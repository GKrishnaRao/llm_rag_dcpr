"""Google Cloud Storage operations — upload, list, and download documents."""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

from config import Config

logger = logging.getLogger(__name__)


class GCSHandler:
    """Handles all interactions with Google Cloud Storage."""

    def __init__(self) -> None:
        self._client: storage.Client | None = None
        self._bucket: storage.Bucket | None = None

    @property
    def client(self) -> storage.Client:
        if self._client is None:
            self._client = storage.Client()
        return self._client

    @property
    def bucket(self) -> storage.Bucket:
        if self._bucket is None:
            self._bucket = self.client.bucket(Config.GCS_BUCKET_NAME)
        return self._bucket

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_file(self, file_obj: BinaryIO, filename: str) -> dict:
        """
        Upload *file_obj* to GCS under the path ``documents/<filename>``.

        Returns a metadata dict with ``gcs_path``, ``filename``, and
        ``uploaded_at``.

        Raises ``ValueError`` if the file extension is not supported.
        Raises ``RuntimeError`` on GCS errors.
        """
        ext = Path(filename).suffix.lower()
        if ext not in Config.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(Config.SUPPORTED_EXTENSIONS)}"
            )

        gcs_path = f"documents/{filename}"
        blob = self.bucket.blob(gcs_path)

        try:
            file_obj.seek(0)
            blob.upload_from_file(file_obj, content_type=self._content_type(ext))
        except GoogleCloudError as exc:
            logger.error("GCS upload failed for %s: %s", filename, exc)
            raise RuntimeError(f"Upload failed: {exc}") from exc

        uploaded_at = datetime.now(tz=timezone.utc).isoformat()
        blob.metadata = {"uploaded_at": uploaded_at}
        blob.patch()

        return {
            "filename": filename,
            "gcs_path": gcs_path,
            "uploaded_at": uploaded_at,
        }

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_documents(self, limit: int = Config.MAX_DOCUMENTS_DISPLAY) -> list[dict]:
        """
        Return the *limit* most recently uploaded documents from GCS.

        Each item has ``filename``, ``gcs_path``, ``uploaded_at``, and
        ``size_kb``.
        """
        try:
            blobs = list(self.client.list_blobs(Config.GCS_BUCKET_NAME, prefix="documents/"))
        except GoogleCloudError as exc:
            logger.error("Failed to list GCS blobs: %s", exc)
            raise RuntimeError(f"Could not list documents: {exc}") from exc

        docs = []
        for blob in blobs:
            if blob.name == "documents/":
                continue
            blob.reload()
            uploaded_at = (
                blob.metadata.get("uploaded_at") if blob.metadata else None
            ) or blob.time_created.isoformat()
            docs.append(
                {
                    "filename": Path(blob.name).name,
                    "gcs_path": blob.name,
                    "uploaded_at": uploaded_at,
                    "size_kb": round(blob.size / 1024, 1) if blob.size else 0,
                }
            )

        docs.sort(key=lambda d: d["uploaded_at"], reverse=True)
        return docs[:limit]

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_file(self, gcs_path: str) -> bytes:
        """Download a blob and return its raw bytes."""
        blob = self.bucket.blob(gcs_path)
        try:
            return blob.download_as_bytes()
        except GoogleCloudError as exc:
            logger.error("Failed to download %s: %s", gcs_path, exc)
            raise RuntimeError(f"Download failed: {exc}") from exc

    def download_as_stream(self, gcs_path: str) -> io.BytesIO:
        """Return a BytesIO stream for the blob content."""
        return io.BytesIO(self.download_file(gcs_path))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_type(ext: str) -> str:
        mapping = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        return mapping.get(ext, "application/octet-stream")
