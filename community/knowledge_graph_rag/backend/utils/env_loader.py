import os
from pathlib import Path
from typing import Optional

SHARED_KEYS = {"MILVUS_COLLECTION_NAME", "LANGSMITH_PROJECT"}


def _load_env_file(
    file_path: Path,
    *,
    override: bool,
    keys: Optional[set[str]] = None,
) -> None:
    if not file_path.exists():
        return

    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            if keys is not None and key not in keys:
                continue
            value = value.strip().strip('"').strip("'")
            if override or key not in os.environ:
                os.environ[key] = value


def load_backend_env() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    project_dir = backend_dir.parent

    backend_env_file = backend_dir / ".env"
    _load_env_file(backend_env_file, override=False)

    shared_env_from_var = os.getenv("KG_RAG_SHARED_ENV")
    shared_env_file = Path(shared_env_from_var) if shared_env_from_var else project_dir / "shared.env"
    _load_env_file(shared_env_file, override=True, keys=SHARED_KEYS)
