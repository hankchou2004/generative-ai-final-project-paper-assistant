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
    # 兩個方法各自的輸出與分數
    pred_fusion: str = ""
    pred_flare:  str = ""
    scores_fusion: dict = field(default_factory=dict)
    scores_flare:  dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
#  資料集載入（streaming，用 n 做 cache key）
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_rows(dataset_name: str, n: int) -> List[dict]:
    from datasets import load_dataset
    if dataset_name == "CShorten/ML-ArXiv-Papers":
        ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train", streaming=True)
        rows = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            rows.append({"paper_id": str(i), "title": row.get("title", ""), "abstract": row.get("abstract", "")})
    else:
        ds = load_dataset("arxiv-community/arxiv_dataset", split="train", streaming=True)
        rows = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            rows.append({"paper_id": row.get("id", str(i)), "title": row.get("title", ""), "abstract": row.get("abstract", "")})
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
            question=SYNTH_QUESTIONS[i % len(SYNTH_QUESTIONS)],
            reference=row.get("abstract", "").strip(),
        )
        for i, row in enumerate(rows)
    ]


# ══════════════════════════════════════════════════════════════════
#  推論（直接以 abstract 為 context，不需 vectorstore）
# ══════════════════════════════════════════════════════════════════

def _prompt(sample: EvalSample, method_note: str) -> str:
    return (
        f"You are a scientific paper assistant. {method_note}\n\n"
        f"Paper Title: {sample.title}\n\n"
        f"Abstract:\n{sample.abstract}\n\n"
        f"Question: {sample.question}\n\n"
        f"Answer concisely based ONLY on the abstract above (2–4 sentences):"
    )


def run_inference(
    samples: List[EvalSample],
    method: str,           # "fusion" | "flare"
    provider: str,
    delay: float,
    progress_bar=None,
    status_text=None,
) -> List[EvalSample]:
    """
    對所有樣本執行推論，寫入 pred_fusion 或 pred_flare。
    FLARE 額外做一輪自我反思：若初稿含 [UNCERTAIN] 則補充說明。
    """
    from llm_helper import get_llm
    from langchain_core.messages import HumanMessage

    llm   = get_llm(temperature=0.0, provider=provider)
    total = len(samples)

    for i, s in enumerate(samples):
        try:
            if method == "fusion":
                resp = llm.invoke([HumanMessage(content=_prompt(s, ""))])
                s.pred_fusion = resp.content.strip()

            else:  # flare
                # Step 1: draft with uncertainty marking
                draft_note = (
                    "If you are uncertain about a statement, prefix that sentence with [UNCERTAIN]. "
                    "Otherwise answer normally."
                )
                draft_resp = llm.invoke([HumanMessage(content=_prompt(s, draft_note))])
                draft = draft_resp.content.strip()

                # Step 2: if uncertain sentences exist, do one refinement pass
                if "[UNCERTAIN]" in draft:
                    refine_prompt = (
                        f"The following draft answer contains [UNCERTAIN] markers.\n"
                        f"Using only the abstract below, revise those sentences and remove all [UNCERTAIN] markers.\n\n"
                        f"Abstract:\n{s.abstract}\n\n"
                        f"Draft:\n{draft}\n\n"
                        f"Revised answer (no [UNCERTAIN] markers):"
                    )
                    refined = llm.invoke([HumanMessage(content=refine_prompt)])
                    s.pred_flare = refined.content.strip()
                else:
                    s.pred_flare = draft

        except Exception as e:
            logger.error(f"[inference:{method}] sample {i}: {e}")
            if method == "fusion":
                s.pred_fusion = f"[ERROR: {e}]"
            else:
                s.pred_flare = f"[ERROR: {e}]"

        if progress_bar:
            progress_bar.progress((i + 1) / total)
        if status_text:
            status_text.caption(f"{i+1}/{total}: {s.title[:60]}…")
        time.sleep(delay)

    return samples


# ══════════════════════════════════════════════════════════════════
#  評估指標
# ══════════════════════════════════════════════════════════════════

def _rouge(pred: str, ref: str) -> dict:
    try:
        from rouge_score import rouge_scorer
        s = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        r = s.score(ref, pred)
        return {"rouge1_f": round(r["rouge1"].fmeasure, 4), "rougeL_f": round(r["rougeL"].fmeasure, 4)}
    except Exception:
        return {"rouge1_f": 0.0, "rougeL_f": 0.0}


def _bleu1(pred: str, ref: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        sf = SmoothingFunction().method1
        return round(sentence_bleu([ref.lower().split()], pred.lower().split(), weights=(1,0,0,0), smoothing_function=sf), 4)
    except Exception:
        return 0.0


def _exact(pred: str, ref: str) -> float:
    return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0


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


def compute_scores(samples: List[EvalSample], use_bertscore: bool) -> List[EvalSample]:
    """計算兩個方法的所有分數，分別寫入 scores_fusion / scores_flare。"""
    for method in ("fusion", "flare"):
        preds = [s.pred_fusion if method == "fusion" else s.pred_flare for s in samples]
        refs  = [s.reference for s in samples]
        bs    = _bertscore_batch(preds, refs) if use_bertscore else [0.0] * len(samples)
        for i, s in enumerate(samples):
            r = _rouge(preds[i], refs[i])
            scores = {**r, "bleu1": _bleu1(preds[i], refs[i]), "exact_match": _exact(preds[i], refs[i]), "bertscore_f1": bs[i]}
            if method == "fusion":
                s.scores_fusion = scores
            else:
                s.scores_flare = scores
    return samples


def _agg(samples: List[EvalSample], method: str) -> dict:
    if not samples:
        return {}
    keys = ["rouge1_f", "rougeL_f", "bleu1", "exact_match", "bertscore_f1"]
    src  = "scores_fusion" if method == "fusion" else "scores_flare"
    return {k: round(sum(getattr(s, src).get(k, 0.0) for s in samples) / len(samples), 4) for k in keys}


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
    with st.expander(t("⚙️ 評估設定", "⚙️ Evaluation Settings"), expanded=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        dataset_choice = c1.selectbox(
            t("資料集", "Dataset"),
            ["CShorten/ML-ArXiv-Papers", "arxiv-community/arxiv_dataset"],
        )
        n_samples = c2.number_input(t("抽取筆數", "Sample Count"), min_value=10, max_value=500, value=50, step=10)
        use_bertscore = c3.checkbox(t("啟用 BERTScore", "Enable BERTScore"), value=True)
        delay = c4.slider(t("推論間隔(s)", "Delay (s)"), 0.0, 5.0, 0.5, 0.1)

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
                rows = _load_rows(dataset_choice, int(n_samples))
                st.session_state["eval_samples"]      = _build_samples(rows)
                st.session_state["eval_results"]      = None
                st.session_state["eval_prev_agg"]     = None   # 清除前次結果
                st.session_state["eval_lock_info"]    = ""
                st.success(t(f"✅ 載入 {len(rows)} 筆", f"✅ Loaded {len(rows)} records"))
            except Exception as e:
                st.error(t(f"❌ {e}", f"❌ {e}"))
        st.rerun()

    # 批次預覽
    samples: Optional[List[EvalSample]] = st.session_state.get("eval_samples")
    if not samples:
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
        # 保留前次 agg 供 delta 比較
        if st.session_state.get("eval_results"):
            prev = st.session_state["eval_results"]
            st.session_state["eval_prev_agg"] = {
                "fusion": _agg(prev, "fusion"),
                "flare":  _agg(prev, "flare"),
            }

        # 深拷貝樣本（清除舊 predictions，保留 idx/question/reference）
        working = copy.deepcopy(samples)
        for s in working:
            s.pred_fusion = s.pred_flare = ""
            s.scores_fusion = s.scores_flare = {}

        prog_fusion = st.progress(0.0, text=t("🔀 RAG Fusion 推論中…", "🔀 RAG Fusion inferring…"))
        stat_fusion = st.empty()

        with st.status(t("🔀 RAG Fusion 推論", "🔀 RAG Fusion Inference"), expanded=False) as st_f:
            try:
                working = run_inference(working, "fusion", provider, delay, prog_fusion, stat_fusion)
                st_f.update(label=t("✅ RAG Fusion 推論完成", "✅ RAG Fusion done"), state="complete")
            except Exception as e:
                st_f.update(label=t(f"❌ {e}", f"❌ {e}"), state="error")
                st.stop()
        prog_fusion.empty(); stat_fusion.empty()

        prog_flare = st.progress(0.0, text=t("⚡ FLARE 推論中…", "⚡ FLARE inferring…"))
        stat_flare = st.empty()

        with st.status(t("⚡ RAG Fusion+FLARE 推論", "⚡ RAG Fusion+FLARE Inference"), expanded=False) as st_fl:
            try:
                working = run_inference(working, "flare", provider, delay, prog_flare, stat_flare)
                st_fl.update(label=t("✅ FLARE 推論完成", "✅ FLARE done"), state="complete")
            except Exception as e:
                st_fl.update(label=t(f"❌ {e}", f"❌ {e}"), state="error")
                st.stop()
        prog_flare.empty(); stat_flare.empty()

        with st.status(t("📊 計算評估指標…", "📊 Computing metrics…"), expanded=False) as st_m:
            working = compute_scores(working, use_bertscore)
            st_m.update(label=t("✅ 評估完成", "✅ Metrics done"), state="complete")

        st.session_state["eval_results"] = working
        st.rerun()

    # ── 結果顯示 ─────────────────────────────────────────────────
    results: Optional[List[EvalSample]] = st.session_state.get("eval_results")
    if not results:
        return

    agg_f  = _agg(results, "fusion")
    agg_fl = _agg(results, "flare")
    prev   = st.session_state.get("eval_prev_agg")  # {"fusion":..., "flare":...} or None

    st.markdown("---")
    st.subheader(t("📊 對比評估結果", "📊 Comparison Results"))

    METRIC_META = [
        ("ROUGE-1 F1", "rouge1_f"),
        ("ROUGE-L F1", "rougeL_f"),
        ("BLEU-1",     "bleu1"),
        ("Exact Match","exact_match"),
        ("BERTScore",  "bertscore_f1"),
    ]

    # 兩欄並排指標卡片
    col_f, col_fl = st.columns(2)
    col_f.markdown(f"#### 🔀 RAG Fusion")
    col_fl.markdown(f"#### ⚡ RAG Fusion + FLARE")

    for label, key in METRIC_META:
        vf  = agg_f.get(key, 0.0)
        vfl = agg_fl.get(key, 0.0)
        # delta vs previous run (same method)
        delta_f  = round(vf  - prev["fusion"].get(key, 0.0), 4) if prev else None
        delta_fl = round(vfl - prev["flare"].get(key, 0.0),  4) if prev else None
        col_f.metric(label,  f"{vf:.4f}",  delta=f"{delta_f:+.4f}"  if delta_f  is not None else None)
        col_fl.metric(label, f"{vfl:.4f}", delta=f"{delta_fl:+.4f}" if delta_fl is not None else None)

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
        for label, key in METRIC_META:
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
            for label, key in METRIC_META:
                line_rows.append({"idx": s.idx, "metric": label, "score": s.scores_fusion.get(key, 0.0), "method": "🔀 RAG Fusion"})
                line_rows.append({"idx": s.idx, "metric": label, "score": s.scores_flare.get(key, 0.0),  "method": "⚡ FLARE"})
        sel_metric = st.selectbox(t("顯示指標", "Metric"), [m for m, _ in METRIC_META])
        line_df = pd.DataFrame([r for r in line_rows if r["metric"] == sel_metric])
        lc = (
            alt.Chart(line_df)
            .mark_line(point=True, opacity=0.8)
            .encode(
                x=alt.X("idx:Q", title=t("樣本編號", "Sample Index")),
                y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title=sel_metric),
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
            for label, key in METRIC_META:
                row[f"F_{label}"]  = s.scores_fusion.get(key, 0.0)
                row[f"FL_{label}"] = s.scores_flare.get(key, 0.0)
            table_rows.append(row)
        tdf = pd.DataFrame(table_rows)
        score_cols = [c for c in tdf.columns if c.startswith(("F_","FL_"))]
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
            with sc2:
                st.caption("⚡ FLARE")
                for k, v in s.scores_flare.items():
                    fl_v = v
                    fu_v = s.scores_fusion.get(k, 0.0)
                    st.metric(k, f"{fl_v:.4f}", delta=f"{fl_v - fu_v:+.4f}")

    with tab_export:
        st.markdown(t("匯出當次推論的完整結果。", "Export the full results of this run."))
        export_data = [
            {
                "idx": s.idx, "paper_id": s.paper_id, "title": s.title,
                "question": s.question, "reference": s.reference,
                "pred_fusion": s.pred_fusion, "scores_fusion": s.scores_fusion,
                "pred_flare":  s.pred_flare,  "scores_flare":  s.scores_flare,
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
            for k, v in s.scores_flare.items():
                row[f"flare_{k}"] = v
            csv_rows.append(row)
        buf = io.StringIO()
        pd.DataFrame(csv_rows).to_csv(buf, index=False)
        st.download_button(
            t("⬇️ 匯出 CSV", "⬇️ Export CSV"),
            data=buf.getvalue().encode(),
            file_name="eval_scores.csv",
            mime="text/csv",
        )