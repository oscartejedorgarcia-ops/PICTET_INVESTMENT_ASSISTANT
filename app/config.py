"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration – values are read from `.env` or the environment."""

    # LLM (Ollama – local)
    ollama_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2"

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    pdf_dir: Path = base_dir / "data" / "unstructured"
    csv_path: Path = base_dir / "data" / "structured" / "equities.xlsx"
    chroma_dir: Path = base_dir / "storage" / "chroma_db"
    sqlite_path: Path = base_dir / "storage" / "sqlite" / "stocks.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
