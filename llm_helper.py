"""
llm_helper.py - LLM 與 RAG 鏈輔助函式 / LLM & RAG Chain Helper

支援多 Provider 切換：Google Gemini / Groq / OpenAI / Ollama (本地 Llama 3)
Supports multiple providers: Google Gemini / Groq / OpenAI / Ollama (local Llama 3)
"""

import os
import logging
import re
from typing import List

logger = logging.getLogger("llm_helper")

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from operator import itemgetter


# ── Provider 設定 / Provider config ──────────────────────────────────────────

# 可選值 / Possible values: "google" | "groq" | "openai" | "ollama"
# 透過環境變數或 Streamlit session_state 控制
# 注意：模組級快取，呼叫端應明確傳入 provider= 以確保即時性
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")

# 各 provider 預設模型（使用函式動態讀取，確保 sidebar 切換後生效）
def _groq_model()   -> str: return os.getenv("GROQ_CHAT_MODEL",   "llama-3.3-70b-versatile")
def _google_model() -> str: return os.getenv("GOOGLE_CHAT_MODEL",  "gemini-2.0-flash")
def _openai_model() -> str: return os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o-mini")
def _ollama_model() -> str: return os.getenv("OLLAMA_CHAT_MODEL",  "llama3")
def _ollama_url()   -> str: return os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")

# 向後相容的模組級常數（僅供外部直接引用，不在內部使用）
GROQ_CHAT_MODEL   = os.getenv("GROQ_CHAT_MODEL",   "llama-3.3-70b-versatile")
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL",  "gemini-2.0-flash")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o-mini")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL",  "llama3")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0, provider: str = None, model: str = None):
    """
    建立 Chat LLM，依 provider 選擇後端。
    Build Chat LLM based on provider selection.

    Args:
        temperature: 生成溫度
        provider: "google" | "groq" | "openai" | "ollama"，None 則讀取環境變數
        model: 模型名稱，None 則使用各 provider 預設值
    """
    p = provider or os.getenv("LLM_PROVIDER", "google")

    if p == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY 未設定。請在 .streamlit/secrets.toml 加入 GROQ_API_KEY。\n"
                "GROQ_API_KEY is not set. Please add it to .streamlit/secrets.toml."
            )
        m = model or _groq_model()
        logger.debug(f"[get_llm] Groq model={m}, temp={temperature}")
        return ChatGroq(model=m, temperature=temperature, api_key=api_key)

    elif p == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 未設定。請在 .streamlit/secrets.toml 加入 OPENAI_API_KEY。\n"
                "OPENAI_API_KEY is not set. Please add it to .streamlit/secrets.toml."
            )
        m = model or _openai_model()
        logger.debug(f"[get_llm] OpenAI model={m}, temp={temperature}")
        return ChatOpenAI(model=m, temperature=temperature, api_key=api_key)

    elif p == "ollama":
        from langchain_ollama import ChatOllama
        m = model or _ollama_model()
        base_url = _ollama_url()
        logger.debug(f"[get_llm] Ollama model={m}, base_url={base_url}, temp={temperature}")
        return ChatOllama(model=m, temperature=temperature, base_url=base_url)

    else:  # google (default)
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY 未設定。請在 .streamlit/secrets.toml 加入 GOOGLE_API_KEY。\n"
                "GOOGLE_API_KEY is not set. Please add it to .streamlit/secrets.toml."
            )
        m = model or _google_model()
        logger.debug(f"[get_llm] Google Gemini model={m}, temp={temperature}")
        return ChatGoogleGenerativeAI(
            model=m,
            temperature=temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )


# ── Embedding factory ─────────────────────────────────────────────────────────

def get_embedding_func(provider: str = None):
    """
    建立 Embedding 函式。
    Build embedding function.

    Google  → GoogleGenerativeAIEmbeddings (gemini-embedding-001)
    Groq    → HuggingFaceEmbeddings (BAAI/bge-m3，本地，免費，支援中文)
    OpenAI  → OpenAIEmbeddings (text-embedding-3-small)
    Ollama  → OllamaEmbeddings (nomic-embed-text，本地)
    """
    p = provider or os.getenv("LLM_PROVIDER", "google")

    if p == "groq":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = os.getenv("HF_EMBED_MODEL", "BAAI/bge-m3")
        logger.debug(f"[get_embedding_func] HuggingFace embedding model={model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif p == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未設定。")
        embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        logger.debug(f"[get_embedding_func] OpenAI embedding model={embed_model}")
        return OpenAIEmbeddings(model=embed_model, api_key=api_key)

    elif p == "ollama":
        from langchain_ollama import OllamaEmbeddings
        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        base_url = _ollama_url()
        logger.debug(f"[get_embedding_func] Ollama embedding model={embed_model}, base_url={base_url}")
        return OllamaEmbeddings(model=embed_model, base_url=base_url)

    else:  # google
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 未設定。")
        logger.debug("[get_embedding_func] Google gemini-embedding-001")
        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_docs(docs) -> str:
    """將文件列表格式化成 XML 字串。 / Format docs list into XML string."""
    res = ""
    for doc in docs:
        escaped = doc.page_content.replace("\n", "\\n")
        res += "<doc>\n"
        res += f"  <content>{escaped}</content>\n"
        for key, val in doc.metadata.items():
            res += f"  <{key}>{val}</{key}>\n"
        res += "</doc>\n"
    logger.debug(f"[format_docs] 格式化 {len(docs)} 份文件，共 {len(res)} 字元")
    return res


def convert_message(m: dict):
    """將字典格式訊息轉換為 LangChain Message 物件。"""
    role = m["role"]
    content = m["content"]
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        raise ValueError(f"未知角色 / Unknown role: {role}")


def _format_chat_history(chat_history: list) -> str:
    parts = []
    for m in chat_history:
        if isinstance(m, HumanMessage):
            parts.append(f"Human: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
        elif isinstance(m, SystemMessage):
            parts.append(f"System: {m.content}")
    return "\n".join(parts)


# ── Prompts ───────────────────────────────────────────────────────────────────

_condense_template = """\
根據以下對話紀錄和後續問題，將後續問題改寫為獨立問題（保持原語言）。
Given the following conversation and a follow-up question, rephrase it as a standalone question (keep the original language).

Chat History:
{chat_history}
Follow Up Input: {input}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_condense_template)

_rag_template = """\
僅根據以下文件內容回答問題，並引用頁碼（如 [p.3]）。若文件中找不到答案，請如實說明。
Answer the question based ONLY on the context below, citing page numbers (e.g., [p.3]). If the answer is not in the context, say so.

{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(_rag_template)


# ── Vector store loader ───────────────────────────────────────────────────────

def get_search_index(
    file_names: List[str],
    index_folder: str = "index",
    provider: str = None,
) -> List[FAISS]:
    """載入多個 FAISS 索引，embedding 依 provider 選擇。"""
    embeddings = get_embedding_func(provider)
    indexes = []
    for file_name in file_names:
        logger.debug(f"[get_search_index] 載入索引: {file_name}")
        try:
            idx = FAISS.load_local(
                folder_path=index_folder,
                index_name=file_name + ".index",
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            indexes.append(idx)
            logger.info(f"[get_search_index] 成功載入: {file_name}")
        except Exception as e:
            logger.error(f"[get_search_index] 無法載入 {file_name}: {e}", exc_info=True)
            raise
    return indexes


# ── RAG Fusion helpers ────────────────────────────────────────────────────────

def reciprocal_rank_fusion(results: List[List], k: int = 60) -> List:
    """倒數排名融合演算法。"""
    from langchain_core.load import dumps, loads
    fused_scores: dict = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)
    reranked = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    logger.debug(f"[reciprocal_rank_fusion] 融合後共 {len(reranked)} 份文件")
    return reranked


def get_search_query_generation_chain(provider: str = None):
    """建立多查詢生成鏈。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個助手，根據單一輸入查詢生成多個相關搜尋查詢。\nYou are an assistant that generates multiple search queries from a single input query."),
        ("human", "請生成與以下查詢相關的 4 個不同搜尋查詢（每行一個）：\nGenerate 4 different search queries related to: {original_query}\nOUTPUT (4 queries, one per line):")
    ])
    return (
        prompt
        | get_llm(temperature=0, provider=provider)
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )


# ── RAG Fusion Chain (multi-file) ─────────────────────────────────────────────

def get_rag_fusion_chain_files(
    file_names: List[str],
    index_folder: str = "index",
    retrieval_cb=None,
    provider: str = None,
):
    """RAG Fusion 鏈（多檔案，使用倒數排名融合）。"""
    p = provider or os.getenv("LLM_PROVIDER", "google")
    logger.info(f"[get_rag_fusion_chain_files] provider={p}, 檔案: {file_names}")
    vectorstores = get_search_index(file_names, index_folder, provider=p)
    query_gen_chain = get_search_query_generation_chain(provider=p)

    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    def retrieve_and_fuse(queries: List[str]) -> str:
        all_docs = []
        for query in queries:
            per_query_docs = []
            for vs in vectorstores:
                per_query_docs.extend(vs.as_retriever(search_kwargs={"k": 5}).invoke(query))
            all_docs.append(per_query_docs)
        fused = reciprocal_rank_fusion(all_docs)
        return format_docs([doc for doc, _ in fused])

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | get_llm(temperature=0, provider=p)
        | StrOutputParser(),
    )

    _context = {
        "context": (
            RunnablePassthrough.assign(original_query=lambda x: x["standalone_question"])
            | query_gen_chain
            | retrieval_cb
            | retrieve_and_fuse
        ),
        "question": lambda x: x["standalone_question"],
    }

    chain = _inputs | _context | ANSWER_PROMPT | get_llm(provider=p)
    logger.info("[get_rag_fusion_chain_files] RAG Fusion 鏈建立完成")
    return chain


# ── RAG Fusion + FLARE Chain (multi-file) ────────────────────────────────────
#
#  FLARE (Forward-Looking Active REtrieval augmented generation) 核心流程：
#  FLARE core flow:
#
#  1. 用 RAG Fusion 取得初步上下文，生成初稿答案（draft）
#  2. 掃描初稿中低信心句子（含 [UNCERTAIN] 標記 或 以 "I'm not sure" 起頭）
#  3. 針對每個低信心句子，以 RAG Fusion 再次檢索更多證據
#  4. 將補充上下文注入，重新生成最終答案
#
#  此實作為「輕量版 FLARE」：
#  - 不依賴 token-level log-probability（LangChain API 限制）
#  - 改以 LLM 顯式標記不確定語句，迭代最多 MAX_FLARE_ITER 輪
# ─────────────────────────────────────────────────────────────────────────────

MAX_FLARE_ITER = 2   # 最多補充檢索輪數

_flare_draft_template = """\
你是一位論文分析助手。請根據以下文件內容，嘗試回答問題。
You are a paper analysis assistant. Answer the question based on the context below.

如果某個論述你不確定或文件中資訊不足，請在該句子**前面**加上標記 [UNCERTAIN]。
If you are uncertain about a statement or the context is insufficient, prefix that sentence with [UNCERTAIN].

{context}

Question: {question}

Draft Answer (mark uncertain sentences with [UNCERTAIN]):"""

_flare_refine_template = """\
你是一位論文分析助手。以下是初稿答案與補充檢索到的新文件。
You are a paper analysis assistant. Below is a draft answer and additional retrieved documents.

請根據新文件修正初稿中標記為 [UNCERTAIN] 的部分，去除 [UNCERTAIN] 標記，輸出最終完整答案。
Revise the [UNCERTAIN] parts using the new context. Remove all [UNCERTAIN] markers and output the final polished answer.

Original Question: {question}

Draft Answer:
{draft}

Additional Context:
{extra_context}

Final Answer (no [UNCERTAIN] markers, cite page numbers [p.X]):"""

_FLARE_DRAFT_PROMPT  = ChatPromptTemplate.from_template(_flare_draft_template)
_FLARE_REFINE_PROMPT = ChatPromptTemplate.from_template(_flare_refine_template)

_UNCERTAIN_RE = re.compile(r"\[UNCERTAIN\](.+?)(?=\[UNCERTAIN\]|$)", re.DOTALL)


def _extract_uncertain_sentences(draft: str) -> List[str]:
    """擷取草稿中所有 [UNCERTAIN] 標記的句子作為補充查詢。"""
    matches = _UNCERTAIN_RE.findall(draft)
    queries = [m.strip()[:200] for m in matches if m.strip()]
    logger.debug(f"[FLARE] 不確定句子數: {len(queries)}")
    return queries


def get_rag_fusion_flare_chain_files(
    file_names: List[str],
    index_folder: str = "index",
    retrieval_cb=None,
    flare_cb=None,
    provider: str = None,
):
    """
    RAG Fusion + FLARE 鏈（多檔案）。
    RAG Fusion + FLARE chain (multi-file).

    Args:
        file_names:   FAISS 索引名稱列表
        index_folder: 索引目錄
        retrieval_cb: 初次檢索回調（顯示 sub-queries）
        flare_cb:     FLARE 補充檢索回調（顯示不確定句子查詢）
        provider:     "google" | "groq"
    """
    p = provider or os.getenv("LLM_PROVIDER", "google")
    logger.info(f"[get_rag_fusion_flare_chain_files] provider={p}, 檔案: {file_names}")
    vectorstores = get_search_index(file_names, index_folder, provider=p)
    query_gen_chain = get_search_query_generation_chain(provider=p)
    llm = get_llm(temperature=0, provider=p)

    if retrieval_cb is None:
        retrieval_cb = lambda x: x
    if flare_cb is None:
        flare_cb = lambda x: x

    def _multi_retrieve_raw(queries: List[str]) -> List:
        """多查詢 RRF 檢索，回傳 Document 列表。"""
        all_docs = []
        for query in queries:
            per_q = []
            for vs in vectorstores:
                per_q.extend(vs.as_retriever(search_kwargs={"k": 5}).invoke(query))
            all_docs.append(per_q)
        fused = reciprocal_rank_fusion(all_docs)
        return [doc for doc, _ in fused]

    def _retrieve_and_fuse_text(queries: List[str]) -> str:
        docs = _multi_retrieve_raw(queries)
        return format_docs(docs)

    def run_flare(inputs: dict) -> str:
        """
        完整 FLARE 推論流程（同步，可 stream 最終答案由呼叫端處理）。
        Returns the final answer string.
        """
        question = inputs["standalone_question"]

        # ── Step 1: RAG Fusion 初次檢索 ───────────────────────
        sub_queries = query_gen_chain.invoke({"original_query": question})
        retrieval_cb(sub_queries)           # 通知 UI 顯示子查詢
        context = _retrieve_and_fuse_text(sub_queries)

        # ── Step 2: 生成草稿（含 [UNCERTAIN] 標記）────────────
        draft_msg = _FLARE_DRAFT_PROMPT.format_messages(
            context=context, question=question
        )
        draft_resp = llm.invoke(draft_msg)
        draft = draft_resp.content.strip()
        logger.debug(f"[FLARE] 初稿 (前 300 字): {draft[:300]}")

        # ── Step 3: 迭代補充檢索 ──────────────────────────────
        for iteration in range(MAX_FLARE_ITER):
            uncertain_sents = _extract_uncertain_sentences(draft)
            if not uncertain_sents:
                logger.info(f"[FLARE] 第 {iteration+1} 輪：無不確定句子，停止迭代")
                break

            logger.info(f"[FLARE] 第 {iteration+1} 輪補充檢索，查詢數: {len(uncertain_sents)}")
            flare_cb(uncertain_sents)       # 通知 UI 顯示補充查詢

            extra_docs = _multi_retrieve_raw(uncertain_sents)
            extra_context = format_docs(extra_docs)

            refine_msg = _FLARE_REFINE_PROMPT.format_messages(
                question=question,
                draft=draft,
                extra_context=extra_context,
            )
            refined_resp = llm.invoke(refine_msg)
            draft = refined_resp.content.strip()
            logger.debug(f"[FLARE] 第 {iteration+1} 輪精煉後 (前 300 字): {draft[:300]}")

        return draft

    # 建立 Runnable 包裝，讓 .stream() 可正常呼叫
    from langchain_core.runnables import RunnableLambda

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | get_llm(temperature=0, provider=p)
        | StrOutputParser(),
    )

    flare_runnable = RunnableLambda(run_flare)
    chain = _inputs | flare_runnable
    logger.info("[get_rag_fusion_flare_chain_files] RAG Fusion + FLARE 鏈建立完成")
    return chain