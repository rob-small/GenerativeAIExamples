import os
from pathlib import Path

from dotenv import load_dotenv


def load_frontend_env() -> None:
    frontend_dir = Path(__file__).resolve().parent
    project_dir = frontend_dir.parent

    env_from_var = os.getenv("KG_RAG_ENV_FILE")
    env_file = Path(env_from_var) if env_from_var else project_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
