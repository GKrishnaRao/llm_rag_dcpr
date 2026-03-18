"""Application configuration — loads settings from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Google Cloud Storage
    GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "")
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Zilliz Cloud (managed Milvus)
    ZILLIZ_CLOUD_URI: str = os.getenv("ZILLIZ_CLOUD_URI", "")   # e.g. https://xxx.zillizcloud.com
    ZILLIZ_CLOUD_TOKEN: str = os.getenv("ZILLIZ_CLOUD_TOKEN", "")  # API key / token
    ZILLIZ_COLLECTION_NAME: str = os.getenv("ZILLIZ_COLLECTION_NAME", "rag_documents")

    # Embedding model (runs locally, no API key needed)
    # Produces 384-dim vectors — matches the default Zilliz collection schema below
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # LLM model (Groq-hosted)
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    # Document processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    # Cosine similarity threshold: scores >= this value are considered relevant (range 0–1)
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))

    # UI
    MAX_DOCUMENTS_DISPLAY: int = int(os.getenv("MAX_DOCUMENTS_DISPLAY", "10"))

    # Supported file types
    SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".txt", ".docx", ".md", ".csv", ".xlsx"]

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of missing required configuration keys."""
        missing = []
        if not cls.GCS_BUCKET_NAME:
            missing.append("GCS_BUCKET_NAME")
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not cls.ZILLIZ_CLOUD_URI:
            missing.append("ZILLIZ_CLOUD_URI")
        if not cls.ZILLIZ_CLOUD_TOKEN:
            missing.append("ZILLIZ_CLOUD_TOKEN")
        return missing
