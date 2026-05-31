"""
eval_page.py - ArXiv 論文推論與評估介面

兩個方法（RAG Fusion / RAG Fusion+FLARE）在相同資料集上對比評估。
支援「鎖定資料批次」，讓多次推論可使用同一批樣本比較。
"""

from __future__ import annotations

import copy
import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import streamlit as st

logger = logging.getLogger("eval_page")


# ══════════════════════════════════════════════════════════════════
#  資料結構
# ══════════════════════════════════════════════════════════════════

@dataclass
class EvalSample:
    idx:       int
    paper_id:  str
    title:     str
    abstract:  str
    question:  str
    reference: str
    # 兩個方法各自的輸出、分數與延遲
    pred_fusion: str = ""
    pred_flare:  str = ""
    scores_fusion:  dict  = field(default_factory=dict)
    scores_flare:   dict  = field(default_factory=dict)
    latency_fusion: float = 0.0   # 秒
    latency_flare:  float = 0.0


# ══════════════════════════════════════════════════════════════════
#  資料集載入（streaming，用 n 做 cache key）
# ══════════════════════════════════════════════════════════════════

def _check_url(url: str) -> tuple[bool, int]:
    """
    對 URL 發一個 HEAD request，回傳 (ok, status_code)。
    ok=True 代表 HTTP 2xx；若連線失敗則回傳 (False, 0)。
    """
    import urllib.request, urllib.error
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, resp.status
    except urllib.error.HTTPError as e:
        return False, e.code
    except Exception:
        return False, 0


@st.cache_data(show_spinner=False)
def _load_rows(dataset_name: str, n: int, scrolls_config: str = "qasper", leval_config: str = "scientific_qa") -> List[dict]:
    """
    統一載入介面，回傳標準化 dict 列表：
      paper_id, title, abstract, question, reference

    Debug 資訊：
      載入失敗時會在 Streamlit 介面顯示：
        - 實際嘗試的 URL
        - HTTP 狀態碼（404 = 路徑不存在，0 = 連線失敗）
        - 完整 Python exception 訊息
    """
    import traceback
    from datasets import load_dataset
    rows = []

    # ── ArXiv 系列 ──────────────────────────────────────────────
    if dataset_name == "CShorten/ML-ArXiv-Papers":
        ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= n:
                break
            rows.append({
                "paper_id": str(i),
                "title":    row.get("title", ""),
                "abstract": row.get("abstract", "").strip(),
                "question": None,   # 由 _build_samples 填合成問題
                "reference": None,
            })

    elif dataset_name == "arxiv-community/arxiv_dataset":
        ds = load_dataset("arxiv-community/arxiv_dataset", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= n:
                break
            rows.append({
                "paper_id": row.get("id", str(i)),
                "title":    row.get("title", ""),
                "abstract": row.get("abstract", "").strip(),
                "question": None,
                "reference": None,
            })

    # ── SCROLLS ─────────────────────────────────────────────────
    # tau/scrolls Hub repo 存的是 ZIP（qasper.zip 等），下載解壓後才有 JSONL。
    # scrolls.py 的 data_url 即：
    #   https://huggingface.co/datasets/tau/scrolls/resolve/main/<config>.zip
    # 解壓後目錄結構：<config>/train.jsonl, <config>/validation.jsonl, <config>/test.jsonl
    elif dataset_name.startswith("tau/scrolls"):
        import io, zipfile, urllib.request, os
        _VALID_SCROLLS = {
            "qasper", "gov_report", "summ_screen_fd",
            "qmsum", "narrative_qa", "quality", "contract_nli",
        }
        if scrolls_config not in _VALID_SCROLLS:
            raise ValueError(
                f"未知的 SCROLLS 子資料集：'{scrolls_config}'。\n"
                f"可用選項：{sorted(_VALID_SCROLLS)}"
            )
        zip_url = (
            f"https://huggingface.co/datasets/tau/scrolls/resolve/main/{scrolls_config}.zip"
        )
        logger.info(f"[SCROLLS] 下載 ZIP config={scrolls_config}，URL={zip_url}")
        try:
            # 先 HEAD 確認 URL 可存取
            ok, status = _check_url(zip_url)
            if not ok:
                raise RuntimeError(
                    f"ZIP URL 回傳 HTTP {status}\n"
                    f"URL：{zip_url}\n"
                    f"{'→ 需要 HuggingFace token：export HF_TOKEN=hf_xxx 後重啟 streamlit' if status == 403 else '→ 路徑不存在，請至 Hub 確認'}"
                )
            # 下載 ZIP 至記憶體，再解壓讀 validation.jsonl
            token = os.environ.get("HF_TOKEN", "")
            req = urllib.request.Request(zip_url)
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=300) as resp:
                zip_bytes = resp.read()

            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                # 找到 validation.jsonl（可能在子目錄內）
                jsonl_name = next(
                    (n for n in zf.namelist() if n.endswith("validation.jsonl")),
                    None,
                )
                if jsonl_name is None:
                    raise RuntimeError(
                        f"ZIP 內找不到 validation.jsonl。\n"
                        f"ZIP 內容：{zf.namelist()[:20]}"
                    )
                with zf.open(jsonl_name) as f:
                    for i, line in enumerate(f):
                        if i >= n:
                            break
                        row = json.loads(line)
                        raw_input = row.get("input", "")
                        parts = raw_input.rsplit("\n\n", 1)
                        context, question = (
                            (parts[0].strip(), parts[1].strip()) if len(parts) == 2
                            else (raw_input.strip(), "")
                        )
                        output = row.get("output", "")
                        if isinstance(output, list):
                            output = output[0] if output else ""
                        rows.append({
                            "paper_id": str(row.get("id", row.get("pid", str(i)))),
                            "title":    f"[SCROLLS/{scrolls_config}] Sample {i}",
                            "abstract": context,
                            "question": question,
                            "reference": output.strip(),
                        })
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"[SCROLLS] 載入失敗：{exc}\n{tb}")
            raise RuntimeError(
                f"❌ SCROLLS（{scrolls_config}）載入失敗\n\n"
                f"**嘗試的 ZIP URL：** `{zip_url}`\n\n"
                f"**錯誤訊息：** `{exc}`\n\n"
                f"**建議排查：**\n"
                f"- 執行 `curl -L -I '{zip_url}'` 確認 HTTP 狀態\n"
                f"- 若 403：設定 `export HF_TOKEN=hf_xxx` 後重啟 streamlit\n"
                f"- 若需要 token 可至 https://huggingface.co/settings/tokens 產生\n\n"
                f"**完整 traceback：**\n```\n{tb}\n```"
            ) from exc

    # ── L-Eval ──────────────────────────────────────────────────
    # L4NLP/LEval Hub repo 目錄結構（已驗證）：
    #   LEval/Generation/<config>.jsonl
    #   LEval/Exam/<config>.jsonl
    # Hub resolve URL：
    #   https://huggingface.co/datasets/L4NLP/LEval/resolve/main/LEval/<Category>/<config>.jsonl
    elif dataset_name.startswith("L4NLP/LEval"):
        _LEVAL_CATEGORY: dict[str, str] = {
            # Generation tasks
            "financial_qa":     "Generation", "gov_report_summ":   "Generation",
            "legal_contract_qa":"Generation", "meeting_summ":      "Generation",
            "multidoc_qa":      "Generation", "narrative_qa":      "Generation",
            "natural_question": "Generation", "news_summ":         "Generation",
            "paper_assistant":  "Generation", "patent_summ":       "Generation",
            "review_summ":      "Generation", "scientific_qa":     "Generation",
            "tv_show_summ":     "Generation",
            # Exam tasks
            "codeU":   "Exam", "coursera": "Exam",
            "gsm100":  "Exam", "quality":  "Exam",
            "sci_fi":  "Exam", "topic_retrieval_longchat": "Exam",
            "tpo":     "Exam",
        }
        if leval_config not in _LEVAL_CATEGORY:
            raise ValueError(
                f"未知的 L-Eval 子資料集：'{leval_config}'。\n"
                f"可用選項：{sorted(_LEVAL_CATEGORY)}"
            )
        category = _LEVAL_CATEGORY[leval_config]
        url = (
            f"https://huggingface.co/datasets/L4NLP/LEval/resolve/main/"
            f"LEval/{category}/{leval_config}.jsonl"
        )
        logger.info(f"[L-Eval] 嘗試以 JSONL 載入 config={leval_config}，URL={url}")
        try:
            ok, status = _check_url(url)
            if not ok:
                raise RuntimeError(
                    f"Hub URL 回傳 HTTP {status}\n"
                    f"URL：{url}\n"
                    f"{'→ 需要 HuggingFace token：export HF_TOKEN=hf_xxx 後重啟 streamlit' if status == 403 else '→ 路徑不存在'}"
                )
            import os
            token = os.environ.get("HF_TOKEN")
            ds = load_dataset(
                "json",
                data_files={"test": url},
                split="test",
                streaming=True,
                token=token,
            )
            for i, row in enumerate(ds):
                if i >= n:
                    break
                context      = row.get("input", "").strip()
                instructions = row.get("instructions", [])
                outputs      = row.get("outputs", [])
                question  = instructions[0] if instructions else ""
                reference = outputs[0]      if outputs      else ""
                if isinstance(reference, list):
                    reference = reference[0] if reference else ""
                rows.append({
                    "paper_id": str(i),
                    "title":    f"[L-Eval/{leval_config}] Sample {i}",
                    "abstract": context,
                    "question": question,
                    "reference": reference.strip(),
                })
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"[L-Eval] 載入失敗：{exc}\n{tb}")
            raise RuntimeError(
                f"❌ L-Eval（{leval_config}）載入失敗\n\n"
                f"**嘗試的 URL：** `{url}`\n\n"
                f"**錯誤訊息：** `{exc}`\n\n"
                f"**建議排查：**\n"
                f"- 執行 `curl -L -I '{url}'` 確認 HTTP 狀態\n"
                f"- 若 403：設定 `export HF_TOKEN=hf_xxx` 後重啟 streamlit\n\n"
                f"**完整 traceback：**\n```\n{tb}\n```"
            ) from exc

    return rows


SYNTH_QUESTIONS = [
    "What is the main contribution of this paper?",
    "What problem does this paper address?",
    "What method or approach is proposed in this paper?",
    "What are the key findings or results of this paper?",
    "What datasets or benchmarks are used in this paper?",
]

def _build_samples(rows: List[dict]) -> List[EvalSample]:
    return [
        EvalSample(
            idx=i,
            paper_id=row.get("paper_id", str(i)),
            title=row.get("title", ""),
            abstract=row.get("abstract", "").strip(),
            # SCROLLS / L-Eval 已有真實問題；ArXiv 系列用合成問題
            question=row.get("question") or SYNTH_QUESTIONS[i % len(SYNTH_QUESTIONS)],
            reference=row.get("reference") or row.get("abstract", "").strip(),
        )
        for i, row in enumerate(rows)
    ]


# ══════════════════════════════════════════════════════════════════
#  推論工具函式
# ══════════════════════════════════════════════════════════════════

ANSWER_LENGTH_HINTS = {
    "short": "1-2 sentences",
    "medium": "2-4 sentences",
    "long": "4-6 sentences",
}


def _answer_length_hint(answer_length: str) -> str:
    return ANSWER_LENGTH_HINTS.get(answer_length, ANSWER_LENGTH_HINTS["medium"])


def _prompt(sample: EvalSample, method_note: str, answer_length: str = "medium") -> str:
    length_hint = _answer_length_hint(answer_length)
    return (
        f"You are a scientific paper assistant. {method_note}\n\n"
        f"Paper Title: {sample.title}\n\n"
        f"Abstract:\n{sample.abstract}\n\n"
        f"Question: {sample.question}\n\n"
        f"Answer concisely based ONLY on the abstract above ({length_hint}):"
    )


def _compute_delay(prev_pred: str, delay_val: float, delay_unit: str) -> float:
    """delay_unit: 'second' | 'token' | 'sentence'"""
    if delay_unit == "second":
        return delay_val
    elif delay_unit == "token":
        n_tokens = len(prev_pred.split()) if prev_pred else 0
        return (delay_val * n_tokens) / 1000.0
    else:  # sentence
        import re as _re
        n_sents = max(1, len([s for s in _re.split(r'[.!?]+', prev_pred) if s.strip()]))
        return delay_val * n_sents


def _sentence_split(text: str) -> List[str]:
    import re as _re
    return [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _vocab_confidence(sentence: str, context: str) -> float:
    """
    句子信心度 = 句子中有 context 依據的 content word 比例。
    比讓 LLM 打分更穩定：學術套語（"this paper presents"）在 context 中
    不會出現，信心度自然低，正好觸發 refine。
    停用詞（stopwords）排除，避免虛詞拉高分數。
    """
    STOPWORDS = {
        "a","an","the","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","to","of","in","for",
        "on","with","at","by","from","this","that","these","those",
        "it","its","we","our","they","their","and","or","but","not",
        "as","so","such","also","more","most","than","then","when",
        "which","who","how","what","where","paper","study","method",
        "approach","result","using","used","based","show","shows",
        "shown","present","presents","propose","proposes","proposed",
    }
    ctx_words = set(context.lower().split()) - STOPWORDS
    if not ctx_words:
        return 1.0
    sent_words = [w for w in sentence.lower().split() if w not in STOPWORDS]
    if not sent_words:
        return 1.0
    supported = sum(1 for w in sent_words if w in ctx_words)
    return round(supported / len(sent_words), 4)


def _mini_retrieve(query_sentence: str, context: str, top_k: int = 3) -> str:
    """
    評估場景的輕量 retriever：
    把 abstract 切成句子，用 TF-IDF cosine similarity 找最相關的 top_k 句
    當作「補充文件」。這模擬了真實 FLARE 中動態查 vector DB 的行為。
    """
    import math
    sentences = _sentence_split(context)
    if not sentences:
        return context

    def tf_idf_vec(text: str, vocab: List[str]) -> List[float]:
        words = text.lower().split()
        tf = {w: words.count(w) / len(words) for w in set(words)} if words else {}
        return [tf.get(w, 0.0) for w in vocab]

    query_words  = query_sentence.lower().split()
    all_texts    = sentences + [query_sentence]
    vocab        = list(set(w for t in all_texts for w in t.lower().split()))
    query_vec    = tf_idf_vec(query_sentence, vocab)
    query_norm   = math.sqrt(sum(x*x for x in query_vec)) or 1.0

    scored = []
    for sent in sentences:
        sv   = tf_idf_vec(sent, vocab)
        dot  = sum(a*b for a, b in zip(query_vec, sv))
        norm = math.sqrt(sum(x*x for x in sv)) or 1.0
        scored.append((dot / (query_norm * norm), sent))

    scored.sort(key=lambda x: -x[0])
    return " ".join(s for _, s in scored[:top_k])


# ══════════════════════════════════════════════════════════════════
#  推論主函式
# ══════════════════════════════════════════════════════════════════

def run_inference(
    samples: List[EvalSample],
    method: str,                       # "fusion" | "flare"
    provider: str,
    delay: float,
    delay_unit: str = "second",        # "second" | "token" | "sentence"
    uncertain_threshold: float = 0.5,  # 僅 FLARE 使用
    flare_top_k: int = 2,
    answer_length: str = "medium",
    temperature: float = 0.0,
    progress_bar=None,
    status_text=None,
) -> List[EvalSample]:
    """
    Fusion：直接以 abstract 為 context 生成答案。

    FLARE（修正後的真實流程）：
      1. 以 abstract 為 context 生成草稿
      2. 用詞彙覆蓋率（vocab confidence）對每個句子打信心分數
         → 比讓 LLM 自評更穩定，不受過度自信影響
      3. 低於 uncertain_threshold 的句子拿去做 mini retrieval
         → 從 abstract 本身用 TF-IDF cosine 找最相關句子（模擬 vector DB 查詢）
      4. 以「原始 abstract + 補充句子」為新 context，重新生成最終答案
         → FLARE 和 Fusion 的 context 不同，輸出才會實質差異
    """
    from llm_helper import get_llm
    from langchain_core.messages import HumanMessage

    llm       = get_llm(temperature=temperature, provider=provider)
    total     = len(samples)
    prev_pred = ""

    for i, s in enumerate(samples):
        t_start = time.perf_counter()
        try:
            if method == "fusion":
                # ── RAG Fusion：直接生成 ──────────────────────────
                resp          = llm.invoke([HumanMessage(content=_prompt(s, "", answer_length))])
                s.pred_fusion = resp.content.strip()
                prev_pred     = s.pred_fusion
                s.latency_fusion = round(time.perf_counter() - t_start, 3)

            else:
                # ── FLARE Step 1：生成草稿 ────────────────────────
                draft_resp = llm.invoke([HumanMessage(content=_prompt(s, "", answer_length))])
                draft      = draft_resp.content.strip()

                # ── FLARE Step 2：詞彙覆蓋率信心評分 ──────────────
                # 不讓 LLM 自評（過度自信問題），改用客觀詞彙比對
                sentences       = _sentence_split(draft)
                low_conf_sents  = []
                for sent in sentences:
                    conf = _vocab_confidence(sent, s.abstract)
                    if conf < uncertain_threshold:
                        low_conf_sents.append(sent)

                if low_conf_sents:
                    # ── FLARE Step 3：動態補充檢索 ────────────────
                    # 從 abstract 用 TF-IDF 找最相關句子，模擬 vector DB 查詢
                    extra_context_parts = []
                    for lcs in low_conf_sents:
                        retrieved = _mini_retrieve(lcs, s.abstract, top_k=flare_top_k)
                        extra_context_parts.append(retrieved)
                    extra_context = " ".join(extra_context_parts)

                    # ── FLARE Step 4：以擴充 context 重新生成 ─────
                    # context = 原始 abstract + 補充句子（與 Fusion 的 context 不同）
                    augmented_context = (
                        f"{s.abstract}\n\n"
                        f"[Additional relevant context retrieved for uncertain statements:]\n"
                        f"{extra_context}"
                    )
                    refine_prompt = (
                        f"You are a scientific paper assistant.\n\n"
                        f"Paper Title: {s.title}\n\n"
                        f"Context (abstract + retrieved supplements):\n{augmented_context}\n\n"
                        f"Previous draft answer:\n{draft}\n\n"
                        f"Question: {s.question}\n\n"
                        f"The draft may contain unsupported statements. "
                        f"Using the full context above, write an improved, faithful answer "
                        f"({_answer_length_hint(answer_length)}):"
                    )
                    refined      = llm.invoke([HumanMessage(content=refine_prompt)])
                    s.pred_flare = refined.content.strip()
                else:
                    # 所有句子信心度都達標，草稿即最終答案
                    s.pred_flare = draft

                prev_pred       = s.pred_flare
                s.latency_flare = round(time.perf_counter() - t_start, 3)

        except Exception as e:
            logger.error(f"[inference:{method}] sample {i}: {e}")
            elapsed = round(time.perf_counter() - t_start, 3)
            if method == "fusion":
                s.pred_fusion    = f"[ERROR: {e}]"
                s.latency_fusion = elapsed
            else:
                s.pred_flare    = f"[ERROR: {e}]"
                s.latency_flare = elapsed
            prev_pred = ""

        if progress_bar:
            progress_bar.progress((i + 1) / total)
        if status_text:
            # 顯示當前樣本的信心狀態（FLARE 才有意義）
            if method == "flare" and not s.pred_flare.startswith("[ERROR"):
                sents = _sentence_split(s.pred_flare)
                min_conf = min((_vocab_confidence(sn, s.abstract) for sn in sents), default=1.0)
                status_text.caption(f"{i+1}/{total}  min_conf={min_conf:.2f}  {s.title[:45]}…")
            else:
                status_text.caption(f"{i+1}/{total}: {s.title[:60]}…")

        wait = _compute_delay(prev_pred, delay, delay_unit)
        if wait > 0:
            time.sleep(wait)

    return samples


# ══════════════════════════════════════════════════════════════════
#  評估指標
# ══════════════════════════════════════════════════════════════════

def _rougeL(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer
        s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return round(s.score(ref, pred)["rougeL"].fmeasure, 4)
    except Exception:
        return 0.0


@st.cache_resource(show_spinner=False)
def _bertscore_fn():
    try:
        from bert_score import score
        return score
    except ImportError:
        return None


def _bertscore_batch(preds: List[str], refs: List[str]) -> List[float]:
    fn = _bertscore_fn()
    if fn is None:
        return [0.0] * len(preds)
    try:
        _, _, F1 = fn(preds, refs, model_type="distilbert-base-uncased", verbose=False, device="cpu")
        return [round(f.item(), 4) for f in F1]
    except Exception as e:
        logger.warning(f"[bertscore] {e}")
        return [0.0] * len(preds)


def _faithfulness(pred: str, context: str) -> float:
    """
    Faithfulness（忠實度）：pred 中有多少 unigram 可在 context 中找到依據。
    = |pred_tokens ∩ context_tokens| / |pred_tokens|
    衡量「回答有多少內容是從論文中來的」，越高代表幻覺越少。
    """
    pred_tokens    = pred.lower().split()
    context_tokens = set(context.lower().split())
    if not pred_tokens:
        return 0.0
    supported = sum(1 for w in pred_tokens if w in context_tokens)
    return round(supported / len(pred_tokens), 4)


def _compression_ratio(pred: str, ref: str) -> float:
    """
    Compression Ratio：pred 字數 / ref 字數。
    < 1.0 代表輸出比參考答案短（更精煉）
    > 1.0 代表輸出比參考答案長（可能過度展開）
    接近 1.0 最理想。
    """
    ref_len  = len(ref.split())
    pred_len = len(pred.split())
    if ref_len == 0:
        return 0.0
    return round(pred_len / ref_len, 4)


def compute_scores(samples: List[EvalSample], use_bertscore: bool) -> List[EvalSample]:
    """計算兩個方法的所有分數，分別寫入 scores_fusion / scores_flare。"""
    for method in ("fusion", "flare"):
        preds = [s.pred_fusion if method == "fusion" else s.pred_flare for s in samples]
        refs  = [s.reference for s in samples]
        bs    = _bertscore_batch(preds, refs) if use_bertscore else [0.0] * len(samples)
        for i, s in enumerate(samples):
            scores = {
                "bertscore_f1":     bs[i],
                "rougeL_f":         _rougeL(preds[i], refs[i]),
                "faithfulness":     _faithfulness(preds[i], s.abstract),
                "compression_ratio": _compression_ratio(preds[i], refs[i]),
            }
            if method == "fusion":
                s.scores_fusion = scores
            else:
                s.scores_flare = scores
    return samples


def _agg(samples: List[EvalSample], method: str) -> dict:
    if not samples:
        return {}
    keys = ["bertscore_f1", "rougeL_f", "faithfulness", "compression_ratio"]
    src  = "scores_fusion" if method == "fusion" else "scores_flare"
    agg  = {k: round(sum(getattr(s, src).get(k, 0.0) for s in samples) / len(samples), 4) for k in keys}
    lat_field = "latency_fusion" if method == "fusion" else "latency_flare"
    valid_lat = [getattr(s, lat_field) for s in samples if getattr(s, lat_field) > 0]
    agg["latency_s"] = round(sum(valid_lat) / len(valid_lat), 3) if valid_lat else 0.0
    return agg


# ══════════════════════════════════════════════════════════════════
#  頁面渲染
# ══════════════════════════════════════════════════════════════════

def render_eval_page(provider: str, t):
    import pandas as pd
    import altair as alt

    st.title(t("🧪 推論與評估", "🧪 Inference & Evaluation"))
    st.caption(t(
        "RAG Fusion 與 RAG Fusion+FLARE 在相同資料集上對比評估。",
        "Side-by-side evaluation of RAG Fusion vs RAG Fusion+FLARE on the same dataset.",
    ))

    # ── 設定區 ──────────────────────────────────────────────────
    DATASET_OPTIONS = [
        "CShorten/ML-ArXiv-Papers",
        "arxiv-community/arxiv_dataset",
        "tau/scrolls",
        "L4NLP/LEval",
    ]
    SCROLLS_CONFIGS = ["qasper", "gov_report", "summ_screen_fd", "qmsum", "narrative_qa", "quality", "contract_nli"]
    LEVAL_CONFIGS   = ["scientific_qa", "paper_assistant", "financial_qa", "gov_report_summ",
                       "legal_contract_qa", "meeting_summ", "multidoc_qa", "narrative_qa",
                       "news_summ", "patent_summ", "review_summ", "tv_show_summ", "coursera"]

    with st.expander(t("⚙️ 評估設定", "⚙️ Evaluation Settings"), expanded=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        dataset_choice = c1.selectbox(t("資料集", "Dataset"), DATASET_OPTIONS)
        n_samples      = c2.number_input(t("抽取筆數", "Sample Count"), min_value=10, max_value=500, value=50, step=10)
        use_bertscore  = c3.checkbox(t("啟用 BERTScore", "Enable BERTScore"), value=True)

        # ── 間隔設定 ────────────────────────────────────────────
        delay_unit = c4.radio(
            t("間隔單位", "Delay Unit"),
            options=["second", "token", "sentence"],
            format_func=lambda x: {"second": "秒 (s)", "token": "ms/token", "sentence": "秒/句"}[x],
            horizontal=False,
            help=t(
                "second：固定等待秒數\n"
                "token：每 token 等待 N 毫秒（依上一筆輸出 token 數計算）\n"
                "sentence：每句等待 N 秒（依上一筆輸出句子數計算）",
                "second: fixed wait\ntoken: N ms per output token\nsentence: N s per output sentence",
            ),
        )
        d1, d2 = st.columns(2)
        if delay_unit == "second":
            delay = d1.slider(t("等待秒數", "Wait (s)"), 0.0, 10.0, 0.5, 0.5)
            d2.caption(t("每筆推論後固定等待此秒數", "Fixed wait after each sample"))
        elif delay_unit == "token":
            delay = d1.slider(t("ms / token", "ms / token"), 0.0, 50.0, 5.0, 1.0)
            d2.caption(t("等待時間 = 上一筆 token 數 × N ms", "Wait = prev tokens × N ms"))
        else:
            delay = d1.slider(t("秒 / 句", "s / sentence"), 0.0, 5.0, 0.3, 0.1)
            d2.caption(t("等待時間 = 上一筆句子數 × N 秒", "Wait = prev sentences × N s"))

        # ── FLARE 信心度門檻（僅 FLARE 使用）───────────────────
        uncertain_threshold = st.slider(
            t("⚡ FLARE 信心度門檻（僅 FLARE 使用）", "⚡ FLARE Confidence Threshold (FLARE only)"),
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help=t(
                "以詞彙覆蓋率衡量句子信心度（不依賴 LLM 自評，更穩定）。\n"
                "0.0：所有句子都視為高信心，不觸發 refine\n"
                "0.5：句子中 < 50% 的實詞有 abstract 依據時觸發 refine\n"
                "1.0：幾乎所有句子都觸發 refine（token 消耗最高）",
                "Confidence = fraction of content words supported by the abstract.\n"
                "0.0: never trigger refine\n"
                "0.5: trigger refine when < 50% of content words are in abstract\n"
                "1.0: almost always trigger refine",
            ),
        )
        if uncertain_threshold == 0.0:
            st.caption(t("🔵 停用 refine：FLARE 與 Fusion 輸出將完全相同", "🔵 Refine disabled: FLARE = Fusion output"))
        elif uncertain_threshold >= 0.8:
            st.caption(t("🔴 激進模式：大量句子將觸發 refine，token 消耗顯著增加", "🔴 Aggressive: high token usage"))
        else:
            st.caption(t(f"🟡 標準模式：詞彙覆蓋率 < {uncertain_threshold:.0%} 的句子觸發 refine",
                         f"🟡 Standard: sentences with vocab coverage < {uncertain_threshold:.0%} trigger refine"))

        # 把所有執行參數存進 session_state，確保 rerun 後 run_btn block 能拿到正確值
        st.session_state["_eval_cfg"] = {
            "delay": delay, "delay_unit": delay_unit,
            "uncertain_threshold": uncertain_threshold,
            "use_bertscore": use_bertscore,
        }

        # 子 config 選擇（僅 SCROLLS / L-Eval 顯示）
        scrolls_config = leval_config = None
        if dataset_choice == "tau/scrolls":
            scrolls_config = st.selectbox(t("SCROLLS 子資料集", "SCROLLS subset"), SCROLLS_CONFIGS,
                                          help="qasper：NLP論文QA；gov_report：政府報告摘要…")
        elif dataset_choice == "L4NLP/LEval":
            leval_config = st.selectbox(t("L-Eval 子資料集", "L-Eval subset"), LEVAL_CONFIGS,
                                        help="scientific_qa：科學QA；paper_assistant：論文摘要助手…")

    # ── 資料批次管理 ─────────────────────────────────────────────
    st.markdown(f"### {t('資料批次', 'Dataset Batch')}")

    locked    = st.session_state.get("eval_locked", False)
    lock_info = st.session_state.get("eval_lock_info", "")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

    load_clicked = btn_col1.button(
        t("📥 載入新批次", "📥 Load New Batch"),
        disabled=locked,
        help=t("鎖定後無法載入新批次，請先解除鎖定", "Unlock first to load a new batch"),
        use_container_width=True,
    )
    lock_clicked = btn_col2.button(
        t("🔓 解除鎖定" if locked else "🔒 鎖定此批次", "🔓 Unlock" if locked else "🔒 Lock This Batch"),
        use_container_width=True,
        type="secondary",
    )
    if lock_info:
        btn_col3.caption(f"{'🔒' if locked else '🔓'} {lock_info}")

    # 鎖定/解除
    if lock_clicked:
        if locked:
            st.session_state["eval_locked"] = False
        else:
            if st.session_state.get("eval_samples"):
                st.session_state["eval_locked"]   = True
                n = len(st.session_state["eval_samples"])
                st.session_state["eval_lock_info"] = t(
                    f"已鎖定 {n} 筆（{dataset_choice}）",
                    f"Locked {n} samples ({dataset_choice})",
                )
            else:
                st.warning(t("請先載入資料再鎖定", "Load data before locking"))
        st.rerun()

    # 載入
    if load_clicked:
        with st.spinner(t("從 HuggingFace 串流載入…", "Streaming from HuggingFace…")):
            try:
                rows = _load_rows(
                    dataset_choice, int(n_samples),
                    scrolls_config=scrolls_config or "qasper",
                    leval_config=leval_config or "scientific_qa",
                )
                st.session_state["eval_samples"]   = _build_samples(rows)
                st.session_state["eval_results"]   = None
                st.session_state["eval_prev_agg"]  = None
                st.session_state["eval_lock_info"] = ""
                st.success(t(f"✅ 載入 {len(rows)} 筆", f"✅ Loaded {len(rows)} records"))
            except Exception as e:
                err_msg = str(e)
                # 若是我們自訂的 RuntimeError，用 markdown 渲染（含 traceback block）
                if err_msg.startswith("❌"):
                    st.error(t("資料集載入失敗，詳情如下：", "Dataset load failed. Details below:"))
                    with st.expander(t("🐛 Debug 詳細資訊", "🐛 Debug Details"), expanded=True):
                        st.markdown(err_msg)
                else:
                    st.error(t(f"❌ 未預期錯誤：{e}", f"❌ Unexpected error: {e}"))
                    with st.expander(t("🐛 Debug 詳細資訊", "🐛 Debug Details"), expanded=False):
                        import traceback
                        st.code(traceback.format_exc(), language="python")
        st.rerun()

    # 批次預覽
    samples: Optional[List[EvalSample]] = st.session_state.get("eval_samples")
    if not samples:
        with st.expander(t("🧭 自動尋找最佳參數", "🧭 Auto Tune Parameters"), expanded=False):
            st.info(t(
                "請先在上方載入資料批次。自動尋參會使用目前載入的同一批樣本進行比較。",
                "Please load a dataset batch first. Auto tune compares configs using the currently loaded samples.",
            ))
        st.info(t("請先載入資料集。", "Please load a dataset first."))
        return

    with st.expander(t(f"📋 資料預覽（{len(samples)} 筆）", f"📋 Preview ({len(samples)} samples)"), expanded=False):
        st.dataframe(pd.DataFrame([
            {t("編號","Idx"): s.idx, t("標題","Title"): s.title[:70], t("問題","Question"): s.question}
            for s in samples[:10]
        ]), use_container_width=True)

    # ── 執行推論 ─────────────────────────────────────────────────
    st.markdown(f"### {t('執行推論與評估', 'Run Inference & Evaluation')}")
    st.caption(t(
        "兩個方法將使用完全相同的樣本（同 question、同 reference）進行推論與評估。",
        "Both methods run on the exact same samples (same questions and references).",
    ))

    run_btn = st.button(
        t("🚀 執行（RAG Fusion + FLARE 同時跑）", "🚀 Run Both Methods"),
        type="primary",
    )

    if run_btn:
        # 從 session_state 取出執行當下的參數（避免 rerun 後 expander 內 widget 回到 default）
        cfg = st.session_state.get("_eval_cfg", {})
        _delay              = cfg.get("delay", 0.5)
        _delay_unit         = cfg.get("delay_unit", "second")
        _uncertain_threshold = cfg.get("uncertain_threshold", 0.5)
        _use_bertscore      = cfg.get("use_bertscore", True)

        st.caption(t(
            f"▶ 執行設定：間隔={_delay}{_delay_unit}，信心門檻={_uncertain_threshold:.2f}，BERTScore={'開' if _use_bertscore else '關'}",
            f"▶ Config: delay={_delay}{_delay_unit}, threshold={_uncertain_threshold:.2f}, BERTScore={'on' if _use_bertscore else 'off'}",
        ))

        # 保留前次 agg 供 delta 比較
        if st.session_state.get("eval_results"):
            prev = st.session_state["eval_results"]
            st.session_state["eval_prev_agg"] = {
                "fusion": _agg(prev, "fusion"),
                "flare":  _agg(prev, "flare"),
            }

        working = copy.deepcopy(samples)
        for s in working:
            s.pred_fusion = s.pred_flare = ""
            s.scores_fusion = s.scores_flare = {}

        prog_fusion = st.progress(0.0, text=t("🔀 RAG Fusion 推論中…", "🔀 RAG Fusion inferring…"))
        stat_fusion = st.empty()

        with st.status(t("🔀 RAG Fusion 推論", "🔀 RAG Fusion Inference"), expanded=False) as st_f:
            try:
                working = run_inference(
                    working, "fusion", provider, _delay,
                    delay_unit=_delay_unit,
                    uncertain_threshold=_uncertain_threshold,
                    progress_bar=prog_fusion, status_text=stat_fusion,
                )
                st_f.update(label=t("✅ RAG Fusion 推論完成", "✅ RAG Fusion done"), state="complete")
            except Exception as e:
                st_f.update(label=t(f"❌ {e}", f"❌ {e}"), state="error")
                st.stop()
        prog_fusion.empty(); stat_fusion.empty()

        prog_flare = st.progress(0.0, text=t("⚡ FLARE 推論中…", "⚡ FLARE inferring…"))
        stat_flare = st.empty()

        with st.status(t("⚡ RAG Fusion+FLARE 推論", "⚡ RAG Fusion+FLARE Inference"), expanded=False) as st_fl:
            try:
                working = run_inference(
                    working, "flare", provider, _delay,
                    delay_unit=_delay_unit,
                    uncertain_threshold=_uncertain_threshold,
                    progress_bar=prog_flare, status_text=stat_flare,
                )
                st_fl.update(label=t("✅ FLARE 推論完成", "✅ FLARE done"), state="complete")
            except Exception as e:
                st_fl.update(label=t(f"❌ {e}", f"❌ {e}"), state="error")
                st.stop()
        prog_flare.empty(); stat_flare.empty()

        with st.status(t("📊 計算評估指標…", "📊 Computing metrics…"), expanded=False) as st_m:
            working = compute_scores(working, _use_bertscore)
            st_m.update(label=t("✅ 評估完成", "✅ Metrics done"), state="complete")

        st.session_state["eval_results"] = working
        st.rerun()

    # ── 自動尋參 ─────────────────────────────────────────────────
    with st.expander(t("🧭 自動尋找最佳參數", "🧭 Auto Tune Parameters"), expanded=False):
        st.caption(t(
            "使用目前載入的同一批樣本，自動比較 RAG Fusion 與多組 FLARE 信心門檻。"
            "此功能不會覆蓋上方手動推論的結果。",
            "Use the currently loaded samples to compare RAG Fusion with multiple FLARE thresholds. "
            "This will not overwrite the manual inference results above.",
        ))

        tune_c0, tune_c1 = st.columns([1, 2])
        tune_methods = tune_c0.multiselect(
            t("調參方法", "Methods"),
            options=["fusion", "flare"],
            default=["fusion", "flare"],
            format_func=lambda x: {
                "fusion": "RAG Fusion",
                "flare": "RAG Fusion + FLARE",
            }[x],
            help=t(
                "可只跑 Fusion、只跑 FLARE，或兩者一起比較。",
                "Run Fusion only, FLARE only, or compare both.",
            ),
        )
        tune_thresholds = tune_c1.multiselect(
            t("FLARE 信心門檻", "FLARE Confidence Thresholds"),
            options=[0.1, 0.3, 0.5, 0.7, 0.85, 0.95],
            default=[0.3, 0.5, 0.7, 0.85],
            disabled="flare" not in tune_methods,
            help=t(
                "每個門檻都會跑一次 FLARE；RAG Fusion 會自動作為 baseline。",
                "Each threshold runs one FLARE trial; RAG Fusion is included as the baseline.",
            ),
        )

        tune_c2, tune_c3, tune_c4, tune_c5 = st.columns([1, 1, 1, 1])
        tune_top_ks = tune_c2.multiselect(
            t("FLARE top_k", "FLARE top_k"),
            options=[1, 2, 3, 5],
            default=[2],
            disabled="flare" not in tune_methods,
            help=t(
                "每個不確定句會補充檢索幾個最相關句子。",
                "How many related sentences to retrieve for each uncertain sentence.",
            ),
        )
        tune_answer_lengths = tune_c3.multiselect(
            t("回答長度", "Answer Length"),
            options=["short", "medium", "long"],
            default=["medium"],
            format_func=lambda x: {
                "short": t("短：1-2 句", "Short: 1-2 sentences"),
                "medium": t("中：2-4 句", "Medium: 2-4 sentences"),
                "long": t("長：4-6 句", "Long: 4-6 sentences"),
            }[x],
        )
        tune_temperatures = tune_c4.multiselect(
            "Temperature",
            options=[0.0, 0.2, 0.5],
            default=[0.0],
            help=t(
                "0.0 較穩定；較高 temperature 可能更有彈性，但變異較大。",
                "0.0 is more deterministic; higher temperatures may be more flexible but less stable.",
            ),
        )
        tune_objective = tune_c5.selectbox(
            t("最佳化目標", "Optimization Objective"),
            options=["balanced", "accuracy", "faithfulness", "speed"],
            format_func=lambda x: {
                "balanced": t("平衡", "Balanced"),
                "accuracy": t("準確度", "Accuracy"),
                "faithfulness": t("忠實度", "Faithfulness"),
                "speed": t("速度", "Speed"),
            }[x],
        )

        tune_c6, tune_c7 = st.columns([1, 2])
        tune_max_samples = tune_c6.number_input(
            t("尋參樣本上限", "Max Tune Samples"),
            min_value=1,
            max_value=len(samples),
            value=min(10, len(samples)),
            step=1,
            help=t(
                "建議先用少量樣本快速尋參，確認後再用完整批次正式評估。",
                "Start with a small sample count for quick tuning, then evaluate the best config on the full batch.",
            ),
        )
        n_fusion = (
            len(tune_answer_lengths) * len(tune_temperatures)
            if "fusion" in tune_methods else 0
        )
        n_flare = (
            len(tune_thresholds) * len(tune_top_ks) * len(tune_answer_lengths) * len(tune_temperatures)
            if "flare" in tune_methods else 0
        )
        total_configs = n_fusion + n_flare
        tune_c7.caption(t(
            f"預計執行 {total_configs} 組設定 × {int(tune_max_samples)} 筆樣本。",
            f"Will run {total_configs} configs × {int(tune_max_samples)} samples.",
        ))

        tune_btn = st.button(
            t("🔎 開始自動尋參", "🔎 Start Auto Tune"),
            type="secondary",
            disabled=(
                not tune_methods
                or not tune_answer_lengths
                or not tune_temperatures
                or ("flare" in tune_methods and (not tune_thresholds or not tune_top_ks))
            ),
            use_container_width=True,
        )

        if tune_btn:
            from auto_tune import run_auto_tune

            cfg = st.session_state.get("_eval_cfg", {})
            _delay = cfg.get("delay", 0.5)
            _delay_unit = cfg.get("delay_unit", "second")
            _use_bertscore = cfg.get("use_bertscore", True)

            tune_progress = st.progress(0.0, text=t("準備自動尋參…", "Preparing auto tune…"))
            tune_status_text = st.empty()

            def _tune_progress(event: dict):
                total = max(event.get("total", 1), 1)
                idx = event.get("index", 1)
                method = event.get("method", "")
                threshold = event.get("threshold")
                top_k = event.get("flare_top_k")
                answer_length = event.get("answer_length")
                temperature = event.get("temperature")
                if method == "fusion":
                    label = f"RAG Fusion length={answer_length}, temp={temperature}"
                else:
                    label = f"FLARE threshold={threshold}, top_k={top_k}, length={answer_length}, temp={temperature}"
                if event.get("event") == "start":
                    tune_progress.progress((idx - 1) / total, text=t(
                        f"執行 {idx}/{total}：{label}",
                        f"Running {idx}/{total}: {label}",
                    ))
                    tune_status_text.caption(t(
                        f"目前設定：{label}",
                        f"Current config: {label}",
                    ))
                else:
                    tune_progress.progress(idx / total, text=t(
                        f"完成 {idx}/{total}：{label}",
                        f"Done {idx}/{total}: {label}",
                    ))

            try:
                with st.status(t("🧭 自動尋參執行中", "🧭 Auto tuning"), expanded=False) as st_tune:
                    leaderboard = run_auto_tune(
                        samples=samples,
                        provider=provider,
                        thresholds=tune_thresholds,
                        methods=tune_methods,
                        flare_top_ks=tune_top_ks,
                        answer_lengths=tune_answer_lengths,
                        temperatures=tune_temperatures,
                        objective=tune_objective,
                        delay=_delay,
                        delay_unit=_delay_unit,
                        use_bertscore=_use_bertscore,
                        max_samples=int(tune_max_samples),
                        progress_callback=_tune_progress,
                    )
                    st.session_state["auto_tune_leaderboard"] = leaderboard
                    st_tune.update(label=t("✅ 自動尋參完成", "✅ Auto tune complete"), state="complete")
            except Exception as e:
                st.session_state["auto_tune_leaderboard"] = []
                st.error(t(f"❌ 自動尋參失敗：{e}", f"❌ Auto tune failed: {e}"))
            finally:
                tune_progress.empty()
                tune_status_text.empty()

        leaderboard = st.session_state.get("auto_tune_leaderboard", [])
        if leaderboard:
            tune_df = pd.DataFrame(leaderboard)
            best = leaderboard[0]
            best_threshold = best.get("uncertain_threshold")
            if best.get("method") == "fusion":
                best_method = (
                    f"RAG Fusion, length={best.get('answer_length')}, "
                    f"temp={best.get('temperature')}"
                )
            else:
                best_method = (
                    f"FLARE threshold={best_threshold}, top_k={best.get('flare_top_k')}, "
                    f"length={best.get('answer_length')}, temp={best.get('temperature')}"
                )
            st.success(t(
                f"推薦最佳設定：{best_method}，Auto Score = {best.get('auto_score', 0.0):.4f}",
                f"Recommended config: {best_method}, Auto Score = {best.get('auto_score', 0.0):.4f}",
            ))
            st.dataframe(tune_df, use_container_width=True, height=260)

            tune_buf = io.StringIO()
            tune_df.to_csv(tune_buf, index=False)
            st.download_button(
                t("⬇️ 匯出自動尋參 CSV", "⬇️ Export Auto Tune CSV"),
                data=tune_buf.getvalue().encode(),
                file_name="auto_tune_leaderboard.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── 結果顯示 ─────────────────────────────────────────────────
    results: Optional[List[EvalSample]] = st.session_state.get("eval_results")
    if not results:
        return

    agg_f  = _agg(results, "fusion")
    agg_fl = _agg(results, "flare")
    prev   = st.session_state.get("eval_prev_agg")  # {"fusion":..., "flare":...} or None

    st.markdown("---")
    st.subheader(t("📊 對比評估結果", "📊 Comparison Results"))

    # 順序：語意 → 術語覆蓋 → 忠實度 → 長度比
    # Compression Ratio 理想值接近 1.0，不適合用 gradient，單獨處理
    METRIC_META = [
        ("BERTScore F1",       "bertscore_f1",      True),   # (label, key, use_gradient)
        ("ROUGE-L F1",         "rougeL_f",          True),
        ("Faithfulness",       "faithfulness",       True),
        ("Compression Ratio",  "compression_ratio",  False),
    ]

    # 兩欄並排指標卡片
    col_f, col_fl = st.columns(2)
    col_f.markdown(f"#### 🔀 RAG Fusion")
    col_fl.markdown(f"#### ⚡ RAG Fusion + FLARE")

    for label, key, _ in METRIC_META:
        vf  = agg_f.get(key, 0.0)
        vfl = agg_fl.get(key, 0.0)
        delta_f  = round(vf  - prev["fusion"].get(key, 0.0), 4) if prev else None
        delta_fl = round(vfl - prev["flare"].get(key, 0.0),  4) if prev else None
        # Compression Ratio：越接近 1.0 越好，delta_color 用 off（中性顯示）
        dc = "off" if key == "compression_ratio" else "normal"
        col_f.metric(label,  f"{vf:.4f}",  delta=f"{delta_f:+.4f}"  if delta_f  is not None else None, delta_color=dc)
        col_fl.metric(label, f"{vfl:.4f}", delta=f"{delta_fl:+.4f}" if delta_fl is not None else None, delta_color=dc)

    # Latency 獨立一行（delta 負值才是進步，用 delta_color="inverse"）
    lf  = agg_f.get("latency_s", 0.0)
    lfl = agg_fl.get("latency_s", 0.0)
    delta_lf  = round(lf  - prev["fusion"].get("latency_s", 0.0), 3) if prev else None
    delta_lfl = round(lfl - prev["flare"].get("latency_s", 0.0),  3) if prev else None
    col_f.metric(
        t("Latency 平均 (s)", "Avg Latency (s)"),
        f"{lf:.3f}s",
        delta=f"{delta_lf:+.3f}s" if delta_lf is not None else None,
        delta_color="inverse",
    )
    col_fl.metric(
        t("Latency 平均 (s)", "Avg Latency (s)"),
        f"{lfl:.3f}s",
        delta=f"{delta_lfl:+.3f}s" if delta_lfl is not None else None,
        delta_color="inverse",
    )

    if prev:
        st.caption(t("↕️ delta 為與上一次推論結果的差異", "↕️ delta is vs. the previous run"))

    # 頁籤
    tab_bar, tab_line, tab_table, tab_detail, tab_export = st.tabs([
        t("📊 平均分數對比", "📊 Avg Score Comparison"),
        t("📈 逐筆分佈", "📈 Per-Sample Distribution"),
        t("📋 詳細表格", "📋 Detailed Table"),
        t("🔍 逐筆檢視", "🔍 Sample Inspection"),
        t("⬇️ 匯出", "⬇️ Export"),
    ])

    with tab_bar:
        bar_rows = []
        for label, key, _ in METRIC_META:
            bar_rows.append({"metric": label, "score": agg_f.get(key, 0.0),  "method": "🔀 RAG Fusion"})
            bar_rows.append({"metric": label, "score": agg_fl.get(key, 0.0), "method": "⚡ FLARE"})
        bar_df = pd.DataFrame(bar_rows)
        chart = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X("method:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title=t("分數", "Score")),
                color=alt.Color("method:N", scale=alt.Scale(range=["#1a73e8", "#e8711a"]), legend=None),
                column=alt.Column("metric:N", title=None),
                tooltip=["method", "metric", alt.Tooltip("score:Q", format=".4f")],
            )
            .properties(height=260, width=80)
        )
        st.altair_chart(chart)

    with tab_line:
        line_rows = []
        for s in results:
            for label, key, _ in METRIC_META:
                line_rows.append({"idx": s.idx, "metric": label, "score": s.scores_fusion.get(key, 0.0), "method": "🔀 RAG Fusion"})
                line_rows.append({"idx": s.idx, "metric": label, "score": s.scores_flare.get(key, 0.0),  "method": "⚡ FLARE"})
        sel_metric = st.selectbox(t("顯示指標", "Metric"), [m for m, _, _ in METRIC_META])
        line_df = pd.DataFrame([r for r in line_rows if r["metric"] == sel_metric])
        # Compression Ratio y 軸不限 0–1
        y_scale = alt.Scale() if sel_metric == "Compression Ratio" else alt.Scale(domain=[0, 1])
        lc = (
            alt.Chart(line_df)
            .mark_line(point=True, opacity=0.8)
            .encode(
                x=alt.X("idx:Q", title=t("樣本編號", "Sample Index")),
                y=alt.Y("score:Q", scale=y_scale, title=sel_metric),
                color=alt.Color("method:N", scale=alt.Scale(range=["#1a73e8", "#e8711a"])),
                tooltip=["idx", "method", alt.Tooltip("score:Q", format=".4f")],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(lc, use_container_width=True)

    with tab_table:
        table_rows = []
        for s in results:
            row = {t("編號","Idx"): s.idx, t("標題","Title"): s.title[:50]}
            for label, key, _ in METRIC_META:
                row[f"F_{label}"]  = s.scores_fusion.get(key, 0.0)
                row[f"FL_{label}"] = s.scores_flare.get(key, 0.0)
            row["F_Latency(s)"]  = s.latency_fusion
            row["FL_Latency(s)"] = s.latency_flare
            table_rows.append(row)
        tdf = pd.DataFrame(table_rows)
        # gradient 只套用在 use_gradient=True 的指標，排除 Compression Ratio 和 Latency
        gradient_labels = {label for label, _, use_grad in METRIC_META if use_grad}
        score_cols = [c for c in tdf.columns
                      if c.startswith(("F_","FL_"))
                      and "Latency" not in c
                      and any(c[2:] == lbl or c[3:] == lbl for lbl in gradient_labels)]
        st.dataframe(
            tdf.style.background_gradient(subset=score_cols, cmap="YlGn", vmin=0, vmax=1),
            use_container_width=True,
            height=420,
        )

    with tab_detail:
        sidx = st.selectbox(
            t("樣本", "Sample"),
            options=[s.idx for s in results],
            format_func=lambda i: f"#{i}: {results[i].title[:55]}",
        )
        if sidx is not None:
            s = results[sidx]
            st.markdown(f"**{t('標題','Title')}:** {s.title}")
            st.markdown(f"**{t('問題','Question')}:** {s.question}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**{t('Reference','Reference')}**")
                st.text_area("ref_d", s.reference, height=180, label_visibility="collapsed", disabled=True)
            with c2:
                st.markdown("**🔀 RAG Fusion**")
                st.text_area("pred_f_d", s.pred_fusion, height=180, label_visibility="collapsed", disabled=True)
            with c3:
                st.markdown("**⚡ FLARE**")
                st.text_area("pred_fl_d", s.pred_flare, height=180, label_visibility="collapsed", disabled=True)
            st.markdown(f"**{t('分數對比','Score Comparison')}**")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.caption("🔀 RAG Fusion")
                for k, v in s.scores_fusion.items():
                    st.metric(k, f"{v:.4f}")
                st.metric(t("Latency (s)", "Latency (s)"), f"{s.latency_fusion:.3f}s")
            with sc2:
                st.caption("⚡ FLARE")
                for k, v in s.scores_flare.items():
                    fu_v = s.scores_fusion.get(k, 0.0)
                    st.metric(k, f"{v:.4f}", delta=f"{v - fu_v:+.4f}")
                st.metric(
                    t("Latency (s)", "Latency (s)"), f"{s.latency_flare:.3f}s",
                    delta=f"{s.latency_flare - s.latency_fusion:+.3f}s",
                    delta_color="inverse",
                )

    with tab_export:
        st.markdown(t("匯出當次推論的完整結果。", "Export the full results of this run."))
        export_data = [
            {
                "idx": s.idx, "paper_id": s.paper_id, "title": s.title,
                "question": s.question, "reference": s.reference,
                "pred_fusion":     s.pred_fusion,  "scores_fusion":  s.scores_fusion,
                "latency_fusion":  s.latency_fusion,
                "pred_flare":      s.pred_flare,   "scores_flare":   s.scores_flare,
                "latency_flare":   s.latency_flare,
            }
            for s in results
        ]
        st.download_button(
            t("⬇️ 匯出 JSON", "⬇️ Export JSON"),
            data=json.dumps(export_data, ensure_ascii=False, indent=2).encode(),
            file_name="eval_results.json",
            mime="application/json",
        )

        csv_rows = []
        for s in results:
            row = {"idx": s.idx, "title": s.title, "question": s.question}
            for k, v in s.scores_fusion.items():
                row[f"fusion_{k}"] = v
            row["fusion_latency_s"] = s.latency_fusion
            for k, v in s.scores_flare.items():
                row[f"flare_{k}"] = v
            row["flare_latency_s"] = s.latency_flare
            csv_rows.append(row)
        buf = io.StringIO()
        pd.DataFrame(csv_rows).to_csv(buf, index=False)
        st.download_button(
            t("⬇️ 匯出 CSV", "⬇️ Export CSV"),
            data=buf.getvalue().encode(),
            file_name="eval_scores.csv",
            mime="text/csv",
        )
