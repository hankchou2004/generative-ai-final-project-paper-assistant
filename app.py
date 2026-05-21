"""
app.py - PaperHelper 主應用程式 / Main Application

支援 Google Gemini / Groq 雙 Provider 切換。
Supports switching between Google Gemini and Groq providers.
"""

import base64
import logging
import os
import re

import streamlit as st

# ── Debug logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PaperHelper",
    page_icon="🎓",
    layout="wide",
)

# ── 語言切換 / Language toggle ─────────────────────────────────────────────────
LANG = st.sidebar.selectbox("🌐 Language / 語言", options=["中文", "English"], index=0)

def t(zh: str, en: str) -> str:
    return zh if LANG == "中文" else en

# ── Page navigation ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
_page_labels = {
    "中文": {"chat": "💬 論文問答", "eval": "🧪 推論與評估"},
    "English": {"chat": "💬 Paper Q&A", "eval": "🧪 Inference & Eval"},
}
_pl = _page_labels[LANG]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "chat"

nav_col1, nav_col2 = st.sidebar.columns(2)
if nav_col1.button(
    _pl["chat"],
    use_container_width=True,
    type="primary" if st.session_state["current_page"] == "chat" else "secondary",
):
    st.session_state["current_page"] = "chat"
    st.rerun()

if nav_col2.button(
    _pl["eval"],
    use_container_width=True,
    type="primary" if st.session_state["current_page"] == "eval" else "secondary",
):
    st.session_state["current_page"] = "eval"
    st.rerun()

st.sidebar.markdown("---")

# ── Debug toggle ──────────────────────────────────────────────────────────────
show_debug = st.sidebar.checkbox(t("顯示 Debug 資訊", "Show Debug Info"), value=False)

def debug_log(msg: str):
    logger.debug(msg)
    if show_debug:
        st.sidebar.caption(f"🐛 {msg}")


# ── Provider 切換 / Provider switcher ─────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(f"### ⚙️ {t('AI 後端選擇', 'AI Backend')}")

provider_options = {
    t("🔵 Google Gemini（預設）", "🔵 Google Gemini (Default)"): "google",
    t("🟠 Groq + 本地 Embedding", "🟠 Groq + Local Embedding"): "groq",
}
chosen_provider_label = st.sidebar.radio(
    t("選擇 LLM Provider", "Select LLM Provider"),
    list(provider_options.keys()),
    index=0,
)
PROVIDER = provider_options[chosen_provider_label]
os.environ["LLM_PROVIDER"] = PROVIDER
debug_log(f"Provider: {PROVIDER}")

# Provider 說明
if PROVIDER == "groq":
    st.sidebar.info(
        t(
            "**Groq 模式**\n- Chat：Llama 3.3 70B（免費、快速）\n- Embedding：BAAI/bge-m3（本地，支援中文）\n- Vision：不支援，改用 OCR",
            "**Groq Mode**\n- Chat: Llama 3.3 70B (free, fast)\n- Embedding: BAAI/bge-m3 (local, multilingual)\n- Vision: not supported, uses OCR instead",
        )
    )
else:
    st.sidebar.info(
        t(
            "**Google 模式**\n- Chat：Gemini 2.0 Flash\n- Embedding：gemini-embedding-001\n- Vision：Gemini 1.5 Flash",
            "**Google Mode**\n- Chat: Gemini 2.0 Flash\n- Embedding: gemini-embedding-001\n- Vision: Gemini 1.5 Flash",
        )
    )

# Groq 模型選擇
if PROVIDER == "groq":
    groq_model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    chosen_groq_model = st.sidebar.selectbox(
        t("Groq 模型", "Groq Model"),
        groq_model_options,
        index=0,
    )
    os.environ["GROQ_CHAT_MODEL"] = chosen_groq_model

st.sidebar.markdown("---")

# ── API Key 設定 ──────────────────────────────────────────────────────────────

# 從 secrets.toml 讀取
secrets_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_path):
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
            debug_log(t("已從 secrets.toml 讀取 GOOGLE_API_KEY", "Loaded GOOGLE_API_KEY from secrets.toml"))
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
            debug_log(t("已從 secrets.toml 讀取 GROQ_API_KEY", "Loaded GROQ_API_KEY from secrets.toml"))
    except Exception as e:
        logger.warning(f"讀取 secrets.toml 失敗: {e}")

# 依 provider 檢查 API key
if PROVIDER == "google" and not os.getenv("GOOGLE_API_KEY", ""):
    st.sidebar.markdown(
        t(
            "### 🔑 請輸入 Google AI Studio API Key\n前往 [aistudio.google.com](https://aistudio.google.com) 取得。",
            "### 🔑 Enter Google AI Studio API Key\nGet one at [aistudio.google.com](https://aistudio.google.com).",
        )
    )
    api_key_input = st.sidebar.text_input(t("Google API Key", "Google API Key"), type="password", placeholder="AIza...")
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.sidebar.success(t("✅ 金鑰已設定", "✅ API key set"))
        st.rerun()
    else:
        st.warning(t("⚠️ 請在左側欄輸入 Google AI Studio API Key", "⚠️ Please enter your Google AI Studio API Key in the sidebar"))
        st.stop()

elif PROVIDER == "groq" and not os.getenv("GROQ_API_KEY", ""):
    st.sidebar.markdown(
        t(
            "### 🔑 請輸入 Groq API Key\n前往 [console.groq.com](https://console.groq.com) 取得免費金鑰。",
            "### 🔑 Enter Groq API Key\nGet a free key at [console.groq.com](https://console.groq.com).",
        )
    )
    groq_key_input = st.sidebar.text_input(t("Groq API Key", "Groq API Key"), type="password", placeholder="gsk_...")
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input
        st.sidebar.success(t("✅ 金鑰已設定", "✅ API key set"))
        st.rerun()
    else:
        st.warning(t("⚠️ 請在左側欄輸入 Groq API Key", "⚠️ Please enter your Groq API Key in the sidebar"))
        st.stop()


# ── Utility functions ─────────────────────────────────────────────────────────

def extract_arxiv_links(readme_contents: str) -> list[str]:
    links = re.findall(r"https://arxiv\.org/abs/[^\s)]+", readme_contents)
    debug_log(t(f"找到 {len(links)} 個 arXiv 連結", f"Found {len(links)} arXiv links"))
    return links


def get_readme_contents(repo_url: str) -> str | None:
    import requests
    user_repo = repo_url.replace("https://github.com/", "")
    api_url = f"https://api.github.com/repos/{user_repo}/contents/README.md"
    try:
        resp = requests.get(api_url, timeout=10)
        if resp.status_code == 200:
            return base64.b64decode(resp.json()["content"]).decode("utf-8")
        st.sidebar.error(t(f"無法取得 README.md（HTTP {resp.status_code}）", f"Unable to fetch README.md (HTTP {resp.status_code})"))
        return None
    except Exception as e:
        st.sidebar.error(t(f"連線錯誤：{e}", f"Connection error: {e}"))
        return None


def download_arxiv_paper(link: str):
    import arxiv_downloader
    debug_log(t(f"下載 arXiv: {link}", f"Downloading arXiv: {link}"))
    try:
        arxiv_id = arxiv_downloader.utils.url_to_id(link)
        arxiv_downloader.utils.download(arxiv_id, "./pdf", False)
        st.sidebar.success(t(f"✅ 已下載: {link}", f"✅ Downloaded: {link}"))
    except Exception as e:
        st.sidebar.error(t(f"❌ 下載失敗 {link}：{e}", f"❌ Failed to download {link}: {e}"))


# ── Sidebar: Document management ──────────────────────────────────────────────

st.sidebar.markdown(f"### 📂 {t('文件管理', 'Document Management')}")

github_link = st.sidebar.text_input(
    t("GitHub 倉庫網址", "GitHub Repository URL"),
    key="github_link",
    placeholder="https://github.com/username/repo",
)
if github_link:
    readme_contents = get_readme_contents(github_link)
    if readme_contents:
        arxiv_links = extract_arxiv_links(readme_contents)
        if arxiv_links:
            for link in arxiv_links:
                download_arxiv_paper(link)
        else:
            st.sidebar.warning(t("README 中沒有找到 arXiv 連結", "No arXiv links found in README"))

arxiv_link_input = st.sidebar.text_input(
    t("arXiv 論文連結", "arXiv Paper Link"),
    placeholder="https://arxiv.org/abs/xxxx.xxxxx",
    key="arxiv_link_input",
)
if st.sidebar.button(t("⬇️ 下載 arXiv 論文", "⬇️ Download arXiv Paper"), key="dl_arxiv"):
    if arxiv_link_input:
        download_arxiv_paper(arxiv_link_input)
    else:
        st.sidebar.warning(t("請先輸入 arXiv 連結", "Please enter an arXiv link"))

# Vision 選項（僅 Google 模式顯示）
use_vision = False
if PROVIDER == "google":
    use_vision = st.sidebar.checkbox(
        t("🔬 啟用圖表視覺理解（Gemini Vision）", "🔬 Enable Figure/Chart Vision (Gemini Vision)"),
        value=True,
        help=t(
            "啟用後，每頁圖表將由 Gemini Vision 自動描述並納入索引。",
            "When enabled, Gemini Vision describes figures/charts on each page.",
        ),
    )
else:
    st.sidebar.caption(t("ℹ️ Groq 模式下 Vision 不可用，圖表將改用 OCR 辨識。", "ℹ️ Vision unavailable in Groq mode; OCR will be used instead."))

if st.sidebar.button(t("🔄 嵌入所有 PDF 文件", "🔄 Embed All PDF Documents"), key="embed_docs"):
    with st.sidebar.status(t("正在嵌入...", "Embedding..."), expanded=True) as status:
        try:
            import embed_pdf
            results = embed_pdf.embed_all_pdf_docs(use_vision=use_vision, provider=PROVIDER)
            for fname, stats in results.items():
                status.markdown(
                    t(
                        f"📄 {fname}："
                        f"{stats['total_pages']} 頁，"
                        f"{stats['visual_elements']} 個視覺元素，"
                        f"{stats['text_chunks']} 個文字 chunks，"
                        f"{stats['visual_chunks']} 個視覺 chunks，"
                        f"共 {stats['total_chunks']} 個 chunks",

                        f"📄 {fname}: "
                        f"{stats['total_pages']} pages, "
                        f"{stats['visual_elements']} visual elements, "
                        f"{stats['text_chunks']} text chunks, "
                        f"{stats['visual_chunks']} visual chunks, "
                        f"{stats['total_chunks']} total chunks",
                    )
                )
            status.update(label=t("✅ 嵌入完成！", "✅ Embedding complete!"), state="complete")
        except Exception as e:
            logger.error(f"[embed_docs] {e}", exc_info=True)
            status.update(label=t(f"❌ 嵌入失敗：{e}", f"❌ Embedding failed: {e}"), state="error")


# ══════════════════════════════════════════════════════════════════
#  頁面路由 / Page routing
# ══════════════════════════════════════════════════════════════════

CURRENT_PAGE = st.session_state.get("current_page", "chat")

# ── 評估頁面 / Evaluation page ────────────────────────────────────
if CURRENT_PAGE == "eval":
    try:
        import eval_page
        eval_page.render_eval_page(provider=PROVIDER, t=t)
    except Exception as e:
        logger.error(f"[eval_page] 渲染失敗: {e}", exc_info=True)
        st.error(t(f"❌ 評估頁面載入失敗：{e}", f"❌ Evaluation page failed to load: {e}"))
    st.stop()


# ── Main UI (Chat page) ───────────────────────────────────────────────────────

st.title(t("🔎 PaperHelper：高效、精準地閱讀論文", "🔎 PaperHelper: Read Papers Efficiently & Accurately"))
st.caption(
    t(
        f"使用 RAG Fusion · 當前後端：{'Google Gemini' if PROVIDER == 'google' else 'Groq (Llama)'}",
        f"Powered by RAG Fusion · Backend: {'Google Gemini' if PROVIDER == 'google' else 'Groq (Llama)'}",
    )
)

import embed_pdf as embed_pdf_module

try:
    all_index_files = embed_pdf_module.get_all_index_files()
    debug_log(t(f"找到 {len(all_index_files)} 個索引檔案", f"Found {len(all_index_files)} index files"))
except Exception as e:
    all_index_files = []
    debug_log(t(f"讀取索引失敗：{e}", f"Failed to load indexes: {e}"))

chosen_files = st.multiselect(
    t("📄 選擇要搜尋的文件", "📄 Choose files to search"),
    options=all_index_files,
    default=None,
    placeholder=t("請先嵌入文件，再從此選取...", "Embed documents first, then select here..."),
)

# ── RAG method selection ──────────────────────────────────────────────────────

from llm_helper import convert_message, get_rag_chain_files, get_rag_fusion_chain_files

rag_method_map = {
    t("基本 RAG", "Basic RAG"): get_rag_chain_files,
    t("RAG Fusion（推薦）", "RAG Fusion (Recommended)"): get_rag_fusion_chain_files,
}

chosen_rag_method = st.radio(
    t("選擇 RAG 方式", "Choose RAG Method"),
    list(rag_method_map.keys()),
    index=0,
    horizontal=True,
)
get_rag_chain_func = rag_method_map[chosen_rag_method]

if not chosen_files:
    st.info(t(
        "💡 請先在左側欄下載 PDF、點選「嵌入所有 PDF 文件」，再選取文件後開始提問。",
        "💡 Download a PDF via the sidebar, click 'Embed All PDF Documents', select files above, then start chatting.",
    ))

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

prompt = st.chat_input(t("輸入你的問題...", "Enter your question..."))

if prompt:
    if not chosen_files:
        st.warning(t("⚠️ 請先選擇至少一個文件", "⚠️ Please select at least one file first"))
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        retrieval_container = st.container()
        message_placeholder = st.empty()

        retrieval_status = retrieval_container.status(t("**🔍 正在檢索相關內容...**", "**🔍 Retrieving context...**"))
        queried_questions: list[str] = []
        rendered_questions: set[str] = set()

        def update_retrieval_status():
            for q in queried_questions:
                if q not in rendered_questions:
                    rendered_questions.add(q)
                    retrieval_status.markdown(f"\n\n`→ {q}`")

        def retrieval_cb(qs):
            if isinstance(qs, list):
                for q in qs:
                    if q not in queried_questions:
                        queried_questions.append(q)
            return qs

        try:
            custom_chain = get_rag_chain_func(chosen_files, retrieval_cb=retrieval_cb, provider=PROVIDER)
        except Exception as e:
            logger.error(f"[chain build] {e}", exc_info=True)
            st.error(t(f"❌ 建立 RAG 鏈失敗：{e}", f"❌ Failed to build RAG chain: {e}"))
            st.stop()

        chat_history = [convert_message(m) for m in st.session_state.messages[:-1]]

        full_response = ""
        try:
            for chunk in custom_chain.stream({"input": prompt, "chat_history": chat_history}):
                if isinstance(chunk, dict):
                    text = chunk.get("output", "") or chunk.get("content", "")
                elif hasattr(chunk, "content"):
                    text = chunk.content
                else:
                    text = str(chunk)

                full_response += text
                message_placeholder.markdown(full_response + "▌")
                update_retrieval_status()

            retrieval_status.update(label=t("✅ 檢索完成", "✅ Retrieval complete"), state="complete")
            message_placeholder.markdown(full_response)

        except Exception as e:
            logger.error(f"[chain stream] {e}", exc_info=True)
            retrieval_status.update(label=t("❌ 發生錯誤", "❌ Error occurred"), state="error")
            error_msg = t(f"❌ 發生錯誤：{e}", f"❌ Error: {e}")
            message_placeholder.markdown(error_msg)
            full_response = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    t(
        f"PaperHelper · 後端：{'Google Gemini' if PROVIDER == 'google' else 'Groq (Llama)'} + LangChain · 僅依據所選文件回答",
        f"PaperHelper · Backend: {'Google Gemini' if PROVIDER == 'google' else 'Groq (Llama)'} + LangChain · Answers based only on selected documents",
    )
)