import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LANGSMITH_CALLBACKS: Optional[List[Any]] = None
_LANGSMITH_INITIALIZED = False


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def init_langsmith_env() -> bool:
    """Normalize LangSmith/LangChain env vars so tracing works consistently."""
    global _LANGSMITH_INITIALIZED
    if _LANGSMITH_INITIALIZED:
        return _is_truthy(os.getenv("LANGSMITH_TRACING"))

    if _is_truthy(os.getenv("LANGSMITH_TRACING")):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
        if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
        if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
            os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]

    _LANGSMITH_INITIALIZED = True
    return _is_truthy(os.getenv("LANGSMITH_TRACING"))


def get_langsmith_callbacks() -> Optional[List[Any]]:
    """Return LangChain callbacks for LangSmith tracing, or None if disabled."""
    global _LANGSMITH_CALLBACKS
    if not init_langsmith_env():
        return None
    if _LANGSMITH_CALLBACKS is not None:
        return _LANGSMITH_CALLBACKS
    try:
        from langchain.callbacks.tracers import LangChainTracer
    except Exception as exc:
        logger.warning("LangSmith tracing is enabled, but tracer import failed: %s", exc)
        _LANGSMITH_CALLBACKS = None
        return None

    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    _LANGSMITH_CALLBACKS = [LangChainTracer(project_name=project)]
    return _LANGSMITH_CALLBACKS


def get_langsmith_run_config(
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    callbacks = get_langsmith_callbacks()
    if not callbacks and not tags and not metadata:
        return None
    run_config: Dict[str, Any] = {}
    if callbacks:
        run_config["callbacks"] = callbacks
    if tags:
        run_config["tags"] = tags
    if metadata:
        run_config["metadata"] = metadata
    return run_config
