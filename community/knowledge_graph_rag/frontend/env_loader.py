import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

SHARED_KEYS = ("MILVUS_COLLECTION_NAME", "LANGSMITH_PROJECT")


def load_frontend_env() -> None:
    frontend_dir = Path(__file__).resolve().parent
    project_dir = frontend_dir.parent

    load_dotenv(frontend_dir / ".env")

    shared_env_from_var = os.getenv("KG_RAG_SHARED_ENV")
    shared_env_file = Path(shared_env_from_var) if shared_env_from_var else project_dir / "shared.env"
    if not shared_env_file.exists():
        return

    shared_values = dotenv_values(shared_env_file)
    for key in SHARED_KEYS:
        value = shared_values.get(key)
        if value is not None and value != "":
            os.environ[key] = value
