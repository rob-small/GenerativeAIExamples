import os
from pathlib import Path

from dotenv import load_dotenv


def load_frontend_env() -> None:
    project_dir = Path(__file__).resolve().parent.parent

    shared_env_from_var = os.getenv("KG_RAG_SHARED_ENV")
    shared_env_file = Path(shared_env_from_var) if shared_env_from_var else project_dir / "shared.env"
    if shared_env_file.exists():
        # Version-managed defaults.
        load_dotenv(shared_env_file, override=False)

    env_from_var = os.getenv("KG_RAG_ENV_FILE")
    env_file = Path(env_from_var) if env_from_var else project_dir / ".env"
    if env_file.exists():
        # Local, non-versioned sensitive values.
        load_dotenv(env_file, override=True)
