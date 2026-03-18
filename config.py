"""Application configuration — reads from st.secrets (Streamlit Cloud) or .env (local)."""

import os
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    """Return value from st.secrets if running on Streamlit Cloud, else from env."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


class Config:
    # Google Cloud Storage
    GCS_BUCKET_NAME: str = _get("GCS_BUCKET_NAME")
    GOOGLE_APPLICATION_CREDENTIALS: str = _get("GOOGLE_APPLICATION_CREDENTIALS")

    # Groq
    GROQ_API_KEY: str = _get("GROQ_API_KEY")

    # Zilliz Cloud (managed Milvus)
    ZILLIZ_CLOUD_URI: str = _get("ZILLIZ_CLOUD_URI")
    ZILLIZ_CLOUD_TOKEN: str = _get("ZILLIZ_CLOUD_TOKEN")
    ZILLIZ_COLLECTION_NAME: str = _get("ZILLIZ_COLLECTION_NAME", "rag_documents")

    # Embedding model (runs locally, no API key needed)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # LLM model (Groq-hosted)
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    # Document processing
    CHUNK_SIZE: int = int(_get("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP", "200"))

    # Retrieval
    TOP_K_RESULTS: int = int(_get("TOP_K_RESULTS", "5"))
    RELEVANCE_THRESHOLD: float = float(_get("RELEVANCE_THRESHOLD", "0.4"))

    # UI
    MAX_DOCUMENTS_DISPLAY: int = int(_get("MAX_DOCUMENTS_DISPLAY", "10"))

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
