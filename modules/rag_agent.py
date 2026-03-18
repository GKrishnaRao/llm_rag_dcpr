"""RAG agent — retrieves relevant document chunks and generates answers with Claude."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from groq import Groq
from langchain_core.documents import Document

from config import Config
from modules.search_handler import SearchHandler
from modules.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Zilliz returns COSINE similarity scores in [0, 1]; *higher* = more relevant.
# Chunks with score >= RELEVANCE_THRESHOLD are used; the rest trigger web fallback.
_RELEVANCE_THRESHOLD = Config.RELEVANCE_THRESHOLD

_SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant. Answer questions using the \
following priority order:

1. **Document excerpts** (if provided and relevant) — answer from these first and cite the source.
2. **Web search results** (if provided) — use these to give an up-to-date, factual answer. \
Always summarise the key points clearly.
3. **Your own training knowledge** — if neither documents nor web results are available or \
sufficient, answer from your own knowledge. Clearly state: "Based on my knowledge:" before \
the answer so the user knows it is not from their documents or a live search.

Rules:
- ALWAYS attempt to answer. Never say "I do not know" if you have any relevant knowledge.
- Keep answers concise, factual, and well-structured.
- Use markdown: bullet points, bold headings, tables where helpful.
- If answering from web results, mention the sources briefly.
- Never fabricate specific numbers, dates, or legal figures — if uncertain, say so and \
recommend the user verify with an official source."""


@dataclass
class AgentResponse:
    """Structured response from the RAG agent."""

    answer: str
    source_documents: list[Document] = field(default_factory=list)
    web_results: list[dict] = field(default_factory=list)
    used_web_search: bool = False


class RAGAgent:
    """
    Orchestrates retrieval-augmented generation:
    1. Retrieve relevant chunks from the vector store.
    2. If confidence is low, fall back to DuckDuckGo web search.
    3. Generate a grounded answer with Claude.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._vs = vector_store
        self._search = SearchHandler()
        self._client = Groq(api_key=Config.GROQ_API_KEY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> AgentResponse:
        """
        Answer *question* using the RAG pipeline.

        *conversation_history* is a list of ``{"role": ..., "content": ...}``
        dicts to maintain multi-turn context.
        """
        history = conversation_history or []

        # --- Step 1: retrieve from vector store -----------------------
        scored_docs = self._vs.similarity_search_with_score(question, k=Config.TOP_K_RESULTS)
        # Zilliz COSINE scores: higher is better; keep chunks at or above threshold
        relevant_docs = [doc for doc, score in scored_docs if score >= _RELEVANCE_THRESHOLD]

        # --- Step 2: decide if web search is needed -------------------
        web_results: list[dict] = []
        used_web = False

        if not relevant_docs or self._vs.is_empty:
            logger.info("No relevant docs found; falling back to web search.")
            web_results = self._search.search(question)
            used_web = True

        # --- Step 3: build context string for Claude ------------------
        context_parts: list[str] = []

        if relevant_docs:
            doc_context = self._format_doc_context(relevant_docs)
            context_parts.append(f"**Document excerpts:**\n{doc_context}")

        if web_results:
            web_context = self._format_web_context(web_results)
            context_parts.append(f"**Web search results:**\n{web_context}")

        has_context = bool(context_parts)
        context = "\n\n".join(context_parts) if context_parts else ""

        # --- Step 4: call Groq ---------------------------------------
        if has_context:
            user_message = (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer the question using the context above. If the context is insufficient, "
                "supplement with your own knowledge and say so."
            )
        else:
            user_message = (
                f"Question: {question}\n\n"
                "No documents or web results are available for this question. "
                "Answer from your own knowledge and prefix with 'Based on my knowledge:'"
            )

        messages = [*history, {"role": "user", "content": user_message}]

        try:
            response = self._client.chat.completions.create(
                model=Config.LLM_MODEL,
                max_tokens=2048,
                messages=[{"role": "system", "content": _SYSTEM_PROMPT}, *messages],
            )
            answer_text = response.choices[0].message.content
        except Exception as exc:
            logger.error("Claude API error: %s", exc)
            answer_text = f"I encountered an error while generating a response: {exc}"

        return AgentResponse(
            answer=answer_text,
            source_documents=relevant_docs,
            web_results=web_results,
            used_web_search=used_web,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_doc_context(docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (from '{source}'):\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_web_context(results: list[dict]) -> str:
        parts = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "")
            parts.append(f"[{i}] {title} ({href}):\n{body}")
        return "\n\n".join(parts)
