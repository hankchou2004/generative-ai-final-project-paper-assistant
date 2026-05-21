"""
eval_page.py - ArXiv 論文推論與評估介面 / ArXiv Inference & Evaluation Page

從 HuggingFace 資料集載入論文與 QA 對，使用 LLM 進行推論並計算評估指標。
Loads papers and QA pairs from HuggingFace datasets, runs LLM inference, and computes evaluation metrics.

支援資料集 / Supported datasets:
  - CShorten/ML-ArXiv-Papers   → 論文 abstract + title
  - arxiv-community/arxiv_dataset → 論文摘要（含 id / categories）

評估指標 / Evaluation metrics:
  - ROUGE-1 / ROUGE-L (n-gram overlap)
  - BERTScore F1 (semantic similarity, 使用 distilbert-base-uncased)
  - Exact Match (EM)
  - BLEU-1
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import streamlit as st

logger = logging.getLogger("eval_page")

# ══════════════════════════════════════════════════════════════════
#  資料結構 / Data structures
# ══════════════════════════════════════════════════════════════════

@dataclass
class EvalSample:
    """單一評估樣本。"""
    idx:        int
    paper_id:   str
    title:      str
    abstract:   str
    question:   str        # 合成或真實 QA 問題
    reference:  str        # Ground-truth 答案（或 abstract 本身）
    prediction: str = ""   # LLM 推論輸出
    scores:     dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
#  資料集載入 / Dataset loading
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_ml_arxiv_papers(n: int = 100) -> List[dict]:
    """
    載入 CShorten/ML-ArXiv-Papers 的前 n 筆。
    各筆包含: title, abstract
    """
    from datasets import load_dataset
    logger.info(f"[load_ml_arxiv_papers] 載入 {n} 筆 ML-ArXiv-Papers")
    ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        rows.append({
            "paper_id": str(i),
            "title":    row.get("title", ""),
            "abstract": row.get("abstract", ""),
        })
    return rows


@st.cache_data(show_spinner=False)
def load_arxiv_dataset(n: int = 100) -> List[dict]:
    """
    載入 arxiv-community/arxiv_dataset 的前 n 筆。
    各筆包含: id, title, abstract, categories
    """
    from datasets import load_dataset
    logger.info(f"[load_arxiv_dataset] 載入 {n} 筆 arxiv_dataset")
    ds = load_dataset("arxiv-community/arxiv_dataset", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        rows.append({
            "paper_id":   row.get("id", str(i)),
            "title":      row.get("title", ""),
            "abstract":   row.get("abstract", ""),
            "categories": row.get("categories", ""),
        })
    return rows


# ══════════════════════════════════════════════════════════════════
#  問題合成 / Question synthesis
# ══════════════════════════════════════════════════════════════════

SYNTH_QUESTION_TEMPLATES = [
    "What is the main contribution of this paper?",
    "What problem does this paper address?",
    "What method or approach is proposed in this paper?",
    "What are the key findings or results of this paper?",
    "What datasets or benchmarks are used in this paper?",
]

def _pick_question(idx: int) -> str:
    return SYNTH_QUESTION_TEMPLATES[idx % len(SYNTH_QUESTION_TEMPLATES)]


def build_eval_samples(rows: List[dict], question_mode: str = "synthetic") -> List[EvalSample]:
    """
    從資料列建立評估樣本。
    question_mode: "synthetic" → 使用模板問題，reference = abstract
    """
    samples = []
    for i, row in enumerate(rows):
        q = _pick_question(i)
        ref = row.get("abstract", "").strip()
        samples.append(EvalSample(
            idx=i,
            paper_id=row.get("paper_id", str(i)),
            title=row.get("title", ""),
            abstract=ref,
            question=q,
            reference=ref,
        ))
    return samples


# ══════════════════════════════════════════════════════════════════
#  LLM 推論 / LLM Inference
# ══════════════════════════════════════════════════════════════════

def _build_inference_prompt(sample: EvalSample) -> str:
    return (
        f"You are a scientific paper assistant.\n\n"
        f"Paper Title: {sample.title}\n\n"
        f"Abstract:\n{sample.abstract}\n\n"
        f"Question: {sample.question}\n\n"
        f"Answer concisely based ONLY on the abstract above (2–4 sentences):"
    )


def run_inference_on_samples(
    samples: List[EvalSample],
    provider: str,
    progress_bar=None,
    status_text=None,
    delay_seconds: float = 0.5,
) -> List[EvalSample]:
    """
    對所有樣本執行 LLM 推論，更新 sample.prediction。
    """
    from llm_helper import get_llm
    from langchain_core.messages import HumanMessage

    llm = get_llm(temperature=0.0, provider=provider)
    total = len(samples)

    for i, sample in enumerate(samples):
        prompt_text = _build_inference_prompt(sample)
        try:
            resp = llm.invoke([HumanMessage(content=prompt_text)])
            sample.prediction = resp.content.strip()
        except Exception as e:
            logger.error(f"[inference] sample {i} 失敗: {e}", exc_info=True)
            sample.prediction = f"[ERROR: {e}]"

        if progress_bar is not None:
            progress_bar.progress((i + 1) / total)
        if status_text is not None:
            status_text.caption(f"推論中 {i+1}/{total}：{sample.title[:60]}…")

        time.sleep(delay_seconds)   # 避免 Rate Limit

    return samples


# ══════════════════════════════════════════════════════════════════
#  評估指標 / Evaluation metrics
# ══════════════════════════════════════════════════════════════════

def _rouge_scores(prediction: str, reference: str) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        result = scorer.score(reference, prediction)
        return {
            "rouge1_f": round(result["rouge1"].fmeasure, 4),
            "rougeL_f": round(result["rougeL"].fmeasure, 4),
        }
    except Exception as e:
        logger.warning(f"[rouge] 計算失敗: {e}")
        return {"rouge1_f": 0.0, "rougeL_f": 0.0}


def _bleu1_score(prediction: str, reference: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        ref_tokens  = reference.lower().split()
        pred_tokens = prediction.lower().split()
        sf = SmoothingFunction().method1
        return round(sentence_bleu([ref_tokens], pred_tokens, weights=(1,0,0,0), smoothing_function=sf), 4)
    except Exception as e:
        logger.warning(f"[bleu] 計算失敗: {e}")
        return 0.0


def _exact_match(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0


@st.cache_resource(show_spinner=False)
def _load_bertscore():
    """延遲載入 BERTScore（只做一次）。"""
    try:
        from bert_score import score as bert_score_fn
        return bert_score_fn
    except ImportError:
        return None


def _bertscore_batch(predictions: List[str], references: List[str]) -> List[float]:
    fn = _load_bertscore()
    if fn is None:
        return [0.0] * len(predictions)
    try:
        P, R, F1 = fn(
            predictions, references,
            model_type="distilbert-base-uncased",
            verbose=False,
            device="cpu",
        )
        return [round(f.item(), 4) for f in F1]
    except Exception as e:
        logger.warning(f"[bertscore] 批次計算失敗: {e}")
        return [0.0] * len(predictions)


def compute_all_scores(samples: List[EvalSample], use_bertscore: bool = True) -> List[EvalSample]:
    """
    計算所有評估指標，更新 sample.scores。
    """
    # BERTScore 批次計算（效率佳）
    bert_scores_f1 = []
    if use_bertscore:
        preds = [s.prediction for s in samples]
        refs  = [s.reference  for s in samples]
        bert_scores_f1 = _bertscore_batch(preds, refs)
    else:
        bert_scores_f1 = [0.0] * len(samples)

    for i, sample in enumerate(samples):
        r = _rouge_scores(sample.prediction, sample.reference)
        sample.scores = {
            **r,
            "bleu1":      _bleu1_score(sample.prediction, sample.reference),
            "exact_match": _exact_match(sample.prediction, sample.reference),
            "bertscore_f1": bert_scores_f1[i],
        }

    return samples


def aggregate_scores(samples: List[EvalSample]) -> dict:
    """計算所有樣本各指標的平均值。"""
    if not samples:
        return {}
    keys = list(samples[0].scores.keys())
    return {
        k: round(sum(s.scores.get(k, 0.0) for s in samples) / len(samples), 4)
        for k in keys
    }


# ══════════════════════════════════════════════════════════════════
#  Streamlit 頁面渲染 / Page render
# ══════════════════════════════════════════════════════════════════

def render_eval_page(provider: str, t):
    """
    渲染推論與評估完整頁面。
    t: 翻譯函式 t(zh, en)
    """
    st.title(t("🧪 推論與評估", "🧪 Inference & Evaluation"))
    st.caption(
        t(
            "從 HuggingFace 載入 ArXiv 論文，使用 LLM 推論後計算 ROUGE / BERTScore / BLEU 等評估指標。",
            "Load ArXiv papers from HuggingFace, run LLM inference, and compute ROUGE / BERTScore / BLEU metrics.",
        )
    )

    # ── 設定區 ─────────────────────────────────────────────────
    with st.expander(t("⚙️ 評估設定", "⚙️ Evaluation Settings"), expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            dataset_choice = st.selectbox(
                t("選擇資料集", "Dataset"),
                options=[
                    "CShorten/ML-ArXiv-Papers",
                    "arxiv-community/arxiv_dataset",
                ],
                help=t(
                    "ML-ArXiv-Papers：機器學習論文 (title + abstract)\n"
                    "arxiv_dataset：跨領域論文 (含 categories)",
                    "ML-ArXiv-Papers: ML papers (title + abstract)\n"
                    "arxiv_dataset: multi-domain papers (with categories)",
                ),
            )

        with col2:
            n_samples = st.number_input(
                t("抽取筆數", "Sample Count"),
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help=t("建議 50–150 筆，過多可能導致 Rate Limit", "Recommend 50–150; too many may hit rate limits"),
            )

        with col3:
            use_bertscore = st.checkbox(
                t("啟用 BERTScore", "Enable BERTScore"),
                value=True,
                help=t(
                    "BERTScore 使用 distilbert-base-uncased 計算語意相似度，\n首次執行需下載模型（約 250MB）。",
                    "BERTScore uses distilbert-base-uncased for semantic similarity.\nFirst run downloads ~250MB model.",
                ),
            )

        col4, col5 = st.columns(2)
        with col4:
            inference_delay = st.slider(
                t("推論間隔（秒）", "Inference Delay (s)"),
                min_value=0.0, max_value=5.0, value=0.5, step=0.1,
                help=t("避免 API Rate Limit 的等待時間", "Delay between API calls to avoid rate limits"),
            )
        with col5:
            st.markdown(f"**{t('當前後端', 'Current Backend')}:** `{provider.upper()}`")

    # ── 載入資料集 ──────────────────────────────────────────────
    load_col, run_col, export_col = st.columns([1, 1, 1])

    with load_col:
        if st.button(t("📥 載入資料集", "📥 Load Dataset"), type="secondary", use_container_width=True):
            with st.spinner(t("正在從 HuggingFace 串流載入資料…", "Streaming from HuggingFace…")):
                try:
                    if dataset_choice == "CShorten/ML-ArXiv-Papers":
                        rows = load_ml_arxiv_papers(n=int(n_samples))
                    else:
                        rows = load_arxiv_dataset(n=int(n_samples))

                    st.session_state["eval_rows"]   = rows
                    st.session_state["eval_samples"] = build_eval_samples(rows)
                    st.session_state["eval_results"] = []
                    st.session_state["eval_agg"]     = {}
                    st.success(t(f"✅ 已載入 {len(rows)} 筆資料", f"✅ Loaded {len(rows)} records"))
                    logger.info(f"[eval_page] 載入 {len(rows)} 筆，資料集={dataset_choice}")
                except Exception as e:
                    st.error(t(f"❌ 載入失敗：{e}", f"❌ Load failed: {e}"))
                    logger.error(f"[eval_page] 載入失敗: {e}", exc_info=True)

    # 資料集預覽
    if "eval_samples" in st.session_state and st.session_state["eval_samples"]:
        samples: List[EvalSample] = st.session_state["eval_samples"]

        with st.expander(t(f"📋 資料預覽（前 5 筆，共 {len(samples)} 筆）", f"📋 Data Preview (first 5 of {len(samples)})"), expanded=False):
            import pandas as pd
            preview_df = pd.DataFrame([
                {
                    t("編號", "Idx"): s.idx,
                    t("標題", "Title"): s.title[:80] + ("…" if len(s.title) > 80 else ""),
                    t("問題", "Question"): s.question,
                    t("摘要長度", "Abstract Len"): len(s.abstract),
                }
                for s in samples[:5]
            ])
            st.dataframe(preview_df, use_container_width=True)

        # ── 執行推論 ────────────────────────────────────────────
        with run_col:
            run_btn = st.button(t("🚀 執行推論與評估", "🚀 Run Inference & Eval"), type="primary", use_container_width=True)

        if run_btn:
            progress_bar  = st.progress(0.0)
            status_text   = st.empty()

            # 推論
            with st.status(t("🤖 LLM 推論中…", "🤖 Running LLM inference…"), expanded=True) as inf_status:
                try:
                    st.write(t(f"使用後端：{provider.upper()}，共 {len(samples)} 筆", f"Backend: {provider.upper()}, {len(samples)} samples"))
                    samples = run_inference_on_samples(
                        samples,
                        provider=provider,
                        progress_bar=progress_bar,
                        status_text=status_text,
                        delay_seconds=inference_delay,
                    )
                    st.session_state["eval_samples"] = samples
                    inf_status.update(label=t("✅ 推論完成", "✅ Inference complete"), state="complete")
                except Exception as e:
                    inf_status.update(label=t(f"❌ 推論失敗：{e}", f"❌ Inference failed: {e}"), state="error")
                    logger.error(f"[eval_page] 推論失敗: {e}", exc_info=True)
                    st.stop()

            # 評估
            with st.status(t("📊 計算評估指標…", "📊 Computing evaluation metrics…"), expanded=True) as eval_status:
                try:
                    st.write(t(
                        f"計算 ROUGE / BLEU / Exact Match" + (" / BERTScore" if use_bertscore else ""),
                        f"Computing ROUGE / BLEU / Exact Match" + (" / BERTScore" if use_bertscore else ""),
                    ))
                    samples = compute_all_scores(samples, use_bertscore=use_bertscore)
                    agg     = aggregate_scores(samples)
                    st.session_state["eval_results"] = samples
                    st.session_state["eval_agg"]     = agg
                    eval_status.update(label=t("✅ 評估完成", "✅ Evaluation complete"), state="complete")
                except Exception as e:
                    eval_status.update(label=t(f"❌ 評估失敗：{e}", f"❌ Evaluation failed: {e}"), state="error")
                    logger.error(f"[eval_page] 評估失敗: {e}", exc_info=True)

            status_text.empty()
            progress_bar.empty()

    # ── 結果顯示 ────────────────────────────────────────────────
    if st.session_state.get("eval_agg"):
        agg     = st.session_state["eval_agg"]
        results = st.session_state["eval_results"]

        st.markdown("---")
        st.subheader(t("📊 整體評估結果", "📊 Aggregate Evaluation Results"))

        # 指標卡片
        metric_cols = st.columns(5)
        metric_meta = [
            ("ROUGE-1 F1",     "rouge1_f",      "n-gram 精確率/召回率平衡"),
            ("ROUGE-L F1",     "rougeL_f",      "最長公共子序列"),
            ("BLEU-1",         "bleu1",         "1-gram 精確率（平滑）"),
            ("Exact Match",    "exact_match",   "完全匹配比率"),
            ("BERTScore F1",   "bertscore_f1",  "語意相似度"),
        ]
        for col, (label, key, _help) in zip(metric_cols, metric_meta):
            val = agg.get(key, 0.0)
            col.metric(label=label, value=f"{val:.4f}", help=_help)

        # 分數分佈圖
        import pandas as pd

        scores_df = pd.DataFrame([
            {
                "idx":          s.idx,
                "title":        s.title[:50],
                "rouge1_f":     s.scores.get("rouge1_f", 0),
                "rougeL_f":     s.scores.get("rougeL_f", 0),
                "bleu1":        s.scores.get("bleu1", 0),
                "bertscore_f1": s.scores.get("bertscore_f1", 0),
            }
            for s in results
        ])

        tab_chart, tab_table, tab_detail = st.tabs([
            t("📈 分數分佈", "📈 Score Distribution"),
            t("📋 詳細結果", "📋 Detailed Results"),
            t("🔍 逐筆檢視", "🔍 Sample Inspection"),
        ])

        with tab_chart:
            import altair as alt

            chart_data = scores_df[["idx", "rouge1_f", "rougeL_f", "bleu1", "bertscore_f1"]].melt(
                id_vars="idx", var_name="metric", value_name="score"
            )
            chart = (
                alt.Chart(chart_data)
                .mark_line(point=True, opacity=0.7)
                .encode(
                    x=alt.X("idx:Q", title=t("樣本編號", "Sample Index")),
                    y=alt.Y("score:Q", title=t("分數", "Score"), scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("metric:N", title=t("指標", "Metric")),
                    tooltip=["idx", "metric", alt.Tooltip("score:Q", format=".4f")],
                )
                .properties(
                    title=t("各樣本評估分數", "Per-Sample Evaluation Scores"),
                    height=350,
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

            # 長條圖：平均分數
            avg_df = pd.DataFrame([
                {"metric": k, "avg_score": v}
                for k, v in agg.items()
            ])
            bar = (
                alt.Chart(avg_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("metric:N", title=t("指標", "Metric"), sort=None),
                    y=alt.Y("avg_score:Q", title=t("平均分數", "Avg Score"), scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("metric:N", legend=None),
                    tooltip=["metric", alt.Tooltip("avg_score:Q", format=".4f")],
                )
                .properties(title=t("各指標平均分數", "Average Score per Metric"), height=280)
            )
            st.altair_chart(bar, use_container_width=True)

        with tab_table:
            display_df = scores_df.rename(columns={
                "idx": t("編號", "Idx"),
                "title": t("標題", "Title"),
                "rouge1_f": "ROUGE-1",
                "rougeL_f": "ROUGE-L",
                "bleu1": "BLEU-1",
                "bertscore_f1": "BERTScore",
            })
            st.dataframe(
                display_df.style.background_gradient(
                    subset=["ROUGE-1", "ROUGE-L", "BLEU-1", "BERTScore"],
                    cmap="YlGn", vmin=0, vmax=1
                ),
                use_container_width=True,
                height=400,
            )

        with tab_detail:
            sample_idx = st.selectbox(
                t("選擇樣本編號", "Select Sample Index"),
                options=[s.idx for s in results],
                format_func=lambda i: f"#{i}: {results[i].title[:60]}…" if len(results[i].title) > 60 else f"#{i}: {results[i].title}",
            )
            if sample_idx is not None:
                s = results[sample_idx]
                st.markdown(f"**{t('標題', 'Title')}:** {s.title}")
                st.markdown(f"**{t('問題', 'Question')}:** {s.question}")
                st.markdown(f"**Paper ID:** `{s.paper_id}`")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{t('參考答案（摘要）', 'Reference (Abstract)')}**")
                    st.text_area("ref", s.reference, height=200, label_visibility="collapsed", disabled=True)
                with c2:
                    st.markdown(f"**{t('LLM 推論輸出', 'LLM Prediction')}**")
                    st.text_area("pred", s.prediction, height=200, label_visibility="collapsed", disabled=True)

                st.markdown(f"**{t('評估分數', 'Scores')}**")
                score_cols = st.columns(len(s.scores))
                for col, (k, v) in zip(score_cols, s.scores.items()):
                    col.metric(k, f"{v:.4f}")

        # ── 匯出 ───────────────────────────────────────────────
        with export_col:
            import json, io
            export_data = [
                {
                    "idx":        s.idx,
                    "paper_id":   s.paper_id,
                    "title":      s.title,
                    "question":   s.question,
                    "reference":  s.reference,
                    "prediction": s.prediction,
                    "scores":     s.scores,
                }
                for s in results
            ]
            json_bytes = json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                label=t("⬇️ 匯出 JSON 結果", "⬇️ Export JSON Results"),
                data=json_bytes,
                file_name="eval_results.json",
                mime="application/json",
                use_container_width=True,
            )

            # CSV 匯出
            csv_buf = io.StringIO()
            scores_df.to_csv(csv_buf, index=False)
            st.download_button(
                label=t("⬇️ 匯出 CSV 分數", "⬇️ Export CSV Scores"),
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="eval_scores.csv",
                mime="text/csv",
                use_container_width=True,
            )
