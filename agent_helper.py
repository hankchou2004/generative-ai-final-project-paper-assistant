"""
agent_helper.py - Agent 輔助工具 / Agent Helper Utilities

提供 Streamlit 回調整合與重試裝飾器。
Provides Streamlit callback integration and retry decorators.
"""

import logging
from functools import wraps
from tenacity import retry, wait_exponential, stop_after_attempt

logger = logging.getLogger("agent_helper")

# ── Streamlit callback handler (langchain-community v0.2+) ───────────────────
try:
    from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
    logger.debug("使用 langchain_community.callbacks.streamlit.StreamlitCallbackHandler")
except ImportError:
    # Fallback for older installs
    from streamlit.external.langchain import StreamlitCallbackHandler
    logger.warning("回退至舊版 StreamlitCallbackHandler")


def retry_and_streamlit_callback(st_cb: StreamlitCallbackHandler, tool_name: str):
    """
    裝飾器：為工具函式加上 Streamlit 狀態顯示與自動重試。
    Decorator: adds Streamlit status display and auto-retry to a tool function.

    Args:
        st_cb: StreamlitCallbackHandler 實例，若為 None 則跳過回調。
        tool_name: 顯示在 UI 上的工具名稱。
    """
    if st_cb is None:
        logger.debug(f"[retry_and_streamlit_callback] st_cb 為 None，跳過 UI 回調: {tool_name}")
        return lambda x: x

    def decorator(tool_func):
        @wraps(tool_func)
        def decorated_func(*args, **kwargs):
            logger.debug(f"[{tool_name}] 呼叫工具 args={args}, kwargs={kwargs}")

            # Start a new LLM thought if not already started
            if getattr(st_cb, "_current_thought", None) is None:
                st_cb.on_llm_start({}, [[]])

            args_str = (
                " ".join(str(a) for a in args)
                + " "
                + " ".join(f"{k}=`{v}`" for k, v in kwargs.items())
            ).strip()
            st_cb.on_tool_start({"name": tool_name}, args_str)

            @retry(
                wait=wait_exponential(min=2, max=20),
                stop=stop_after_attempt(5),
                reraise=True,
            )
            def retry_wrapper():
                return tool_func(*args, **kwargs)

            try:
                ret_val = retry_wrapper()
                logger.debug(f"[{tool_name}] 工具回傳 (前 200 字): {str(ret_val)[:200]}")
                st_cb.on_tool_end(str(ret_val))
                return ret_val
            except Exception as e:
                logger.error(f"[{tool_name}] 工具執行失敗: {e}", exc_info=True)
                st_cb.on_tool_error(e)
                raise e

        return decorated_func

    return decorator