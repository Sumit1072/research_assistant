import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env file
load_dotenv()


@dataclass
class Settings:
    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # Other
    FAISS_INDEX_PATH: str | None = os.getenv("FAISS_INDEX_PATH")  # optional path to persist


settings = Settings()
