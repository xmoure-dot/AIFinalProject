from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    chroma_persist_dir: str = "./chroma_db"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 20
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5

    model_config = {"env_file": ".env"}

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()

Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
