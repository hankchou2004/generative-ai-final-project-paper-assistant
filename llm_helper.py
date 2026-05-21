"""
llm_helper.py - LLM 與 RAG 鏈輔助函式 / LLM & RAG Chain Helper

支援多 Provider 切換：Google Gemini / Groq
Supports multiple providers: Google Gemini / Groq
"""

import os
import logging
from typing import List

logger = logging.getLogger("llm_helper")

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from operator import itemgetter


# ── Provider 設定 / Provider config ──────────────────────────────────────────

# 可選值 / Possible values: "google" | "groq"
# 透過環境變數或 Streamlit session_state 控制
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")

# Groq 模型選項（免費，速度快）
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

# Google 模型選項
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.0-flash")


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0, provider: str = None, model: str = None):
    """
    建立 Chat LLM，依 provider 選擇後端。
    Build Chat LLM based on provider selection.

    Args:
        temperature: 生成溫度
        provider: "google" 或 "groq"，None 則讀取 LLM_PROVIDER 環境變數
        model: 模型名稱，None 則使用各 provider 預設值
    """
    p = provider or LLM_PROVIDER

    if p == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY 未設定。請在 .streamlit/secrets.toml 加入 GROQ_API_KEY。\n"
                "GROQ_API_KEY is not set. Please add it to .streamlit/secrets.toml."
            )
        m = model or GROQ_CHAT_MODEL
        logger.debug(f"[get_llm] Groq model={m}, temp={temperature}")
        return ChatGroq(
            model=m,
            temperature=temperature,
            api_key=api_key,
        )

    else:  # google (default)
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY 未設定。請在 .streamlit/secrets.toml 加入 GOOGLE_API_KEY。\n"
                "GOOGLE_API_KEY is not set. Please add it to .streamlit/secrets.toml."
            )
        m = model or GOOGLE_CHAT_MODEL
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
    """
    p = provider or LLM_PROVIDER

    if p == "groq":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = os.getenv("HF_EMBED_MODEL", "BAAI/bge-m3")
        logger.debug(f"[get_embedding_func] HuggingFace embedding model={model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

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


# ── RAG Chain (multi-file) ────────────────────────────────────────────────────

def get_rag_chain_files(
    file_names: List[str],
    index_folder: str = "index",
    retrieval_cb=None,
    provider: str = None,
):
    """基本 RAG 鏈（多檔案）。"""
    p = provider or LLM_PROVIDER
    logger.info(f"[get_rag_chain_files] provider={p}, 檔案: {file_names}")
    vectorstores = get_search_index(file_names, index_folder, provider=p)

    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    def multi_retrieve(query: str) -> str:
        docs = []
        for vs in vectorstores:
            docs.extend(vs.as_retriever(search_kwargs={"k": 5}).invoke(query))
        return format_docs(docs)

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
            itemgetter("standalone_question")
            | RunnablePassthrough(func=retrieval_cb)
            | multi_retrieve
        ),
        "question": lambda x: x["standalone_question"],
    }

    chain = _inputs | _context | ANSWER_PROMPT | get_llm(provider=p)
    logger.info("[get_rag_chain_files] RAG 鏈建立完成")
    return chain


# ── RAG Fusion Chain (multi-file) ─────────────────────────────────────────────

def get_rag_fusion_chain_files(
    file_names: List[str],
    index_folder: str = "index",
    retrieval_cb=None,
    provider: str = None,
):
    """RAG Fusion 鏈（多檔案，使用倒數排名融合）。"""
    p = provider or LLM_PROVIDER
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