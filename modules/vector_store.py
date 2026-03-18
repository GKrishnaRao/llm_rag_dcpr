"""Zilliz Cloud vector store using pymilvus MilvusClient directly."""

from __future__ import annotations

import logging
import uuid

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import MilvusClient, DataType

from config import Config

logger = logging.getLogger(__name__)

_COLLECTION = Config.ZILLIZ_COLLECTION_NAME
_DIM = Config.EMBEDDING_DIM


class VectorStore:
    """
    Zilliz Cloud vector store backed by pymilvus MilvusClient.

    Collection schema:
        id        VARCHAR PK
        vector    FLOAT_VECTOR (384-dim, COSINE)
        text      VARCHAR
        source    VARCHAR
    """

    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self._client: MilvusClient | None = None
        self._collection_has_data: bool = False
        self._connect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return not self._collection_has_data

    def add_documents(self, documents: list[Document]) -> None:
        """Embed *documents* and insert them into Zilliz."""
        if not documents or self._client is None:
            return

        self._ensure_collection()

        texts = [doc.page_content for doc in documents]
        vectors = self._embeddings.embed_documents(texts)

        rows = [
            {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "text": text[:65_535],
                "source": doc.metadata.get("source", "")[:512],
            }
            for text, vec, doc in zip(texts, vectors, documents)
        ]

        self._client.insert(collection_name=_COLLECTION, data=rows)
        self._collection_has_data = True
        logger.info("Inserted %d chunks into '%s'", len(rows), _COLLECTION)

    def similarity_search(self, query: str, k: int = Config.TOP_K_RESULTS) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k)]

    def similarity_search_with_score(
        self, query: str, k: int = Config.TOP_K_RESULTS
    ) -> list[tuple[Document, float]]:
        """Return (Document, cosine_score) pairs; higher score = more relevant."""
        if self._client is None or not self._collection_has_data:
            return []
        try:
            query_vec = self._embeddings.embed_query(query)
            results = self._client.search(
                collection_name=_COLLECTION,
                data=[query_vec],
                limit=k,
                output_fields=["text", "source"],
            )
            pairs = []
            for hit in results[0]:
                doc = Document(
                    page_content=hit["entity"]["text"],
                    metadata={"source": hit["entity"].get("source", "")},
                )
                pairs.append((doc, float(hit["distance"])))
            return pairs
        except Exception as exc:
            logger.warning("Zilliz search failed: %s", exc)
            return []

    def document_indexed(self, filename: str) -> bool:
        if self._client is None:
            return False
        try:
            res = self._client.query(
                collection_name=_COLLECTION,
                filter=f'source == "{filename}"',
                limit=1,
            )
            return len(res) > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        if not Config.ZILLIZ_CLOUD_URI or not Config.ZILLIZ_CLOUD_TOKEN:
            logger.warning("ZILLIZ_CLOUD_URI or ZILLIZ_CLOUD_TOKEN not set.")
            return
        try:
            self._client = MilvusClient(
                uri=Config.ZILLIZ_CLOUD_URI,
                token=Config.ZILLIZ_CLOUD_TOKEN,
            )
            if self._client.has_collection(_COLLECTION):
                stats = self._client.get_collection_stats(_COLLECTION)
                count = int(stats.get("row_count", 0))
                self._collection_has_data = count > 0
                logger.info("Connected to '%s' (rows=%d)", _COLLECTION, count)
            else:
                logger.info("Collection '%s' does not exist — will create on first insert.", _COLLECTION)
        except Exception as exc:
            logger.error("Zilliz connect failed: %s", exc)
            self._client = None

    def _ensure_collection(self) -> None:
        """Create (or recreate) the collection with the required schema."""
        if self._client is None:
            return

        # Drop if schema is incomplete (missing text/source fields)
        if self._client.has_collection(_COLLECTION):
            existing_fields = {
                f["name"]
                for f in self._client.describe_collection(_COLLECTION)["fields"]
            }
            if not {"text", "source"}.issubset(existing_fields):
                logger.warning(
                    "Collection '%s' has incomplete schema %s — dropping and recreating.",
                    _COLLECTION, existing_fields,
                )
                self._client.drop_collection(_COLLECTION)
            else:
                return  # schema already correct

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id",     DataType.VARCHAR, max_length=64,    is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=_DIM)
        schema.add_field("text",   DataType.VARCHAR, max_length=65_535)
        schema.add_field("source", DataType.VARCHAR, max_length=512)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self._client.create_collection(
            collection_name=_COLLECTION,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Created Zilliz collection '%s'", _COLLECTION)
