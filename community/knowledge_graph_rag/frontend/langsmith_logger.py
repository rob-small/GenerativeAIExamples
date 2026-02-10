import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CLIENT = None
_CLIENT_READY = False


def _is_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes", "on"}


def _get_client():
    global _CLIENT, _CLIENT_READY
    if _CLIENT_READY:
        return _CLIENT
    _CLIENT_READY = True
    if not _is_enabled():
        return None
    try:
        from langsmith import Client
    except Exception as exc:
        logger.warning("LangSmith client import failed: %s", exc)
        _CLIENT = None
        return None
    try:
        _CLIENT = Client()
    except Exception as exc:
        logger.warning("LangSmith client init failed: %s", exc)
        _CLIENT = None
    return _CLIENT


def log_event(
    name: str,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    tags: Optional[list] = None,
) -> Optional[str]:
    client = _get_client()
    if not client:
        return None
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    try:
        run = client.create_run(
            name=name,
            run_type="tool",
            inputs=inputs or {},
            outputs=outputs,
            error=error,
            tags=tags,
            project_name=project,
        )
        return run.id if hasattr(run, "id") else None
    except Exception as exc:
        logger.warning("LangSmith logging failed for %s: %s", name, exc)
        return None
