import os
from pathlib import Path

from dotenv import load_dotenv


def load_backend_env() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    project_dir = backend_dir.parent

    env_from_var = os.getenv("KG_RAG_ENV_FILE")
    env_file = Path(env_from_var) if env_from_var else project_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
