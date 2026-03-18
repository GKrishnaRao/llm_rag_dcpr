"""DuckDuckGo web search fallback when the RAG agent cannot answer from documents."""

from __future__ import annotations

import logging

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

_MAX_RESULTS = 5


class SearchHandler:
    """Wraps DuckDuckGo text search."""

    def search(self, query: str, max_results: int = _MAX_RESULTS) -> list[dict]:
        """
        Search DuckDuckGo for *query* and return a list of result dicts.

        Each dict contains ``title``, ``href``, and ``body``.
        Returns an empty list on failure.
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            logger.info("DuckDuckGo returned %d results for: %s", len(results), query)
            return results
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return []

    def format_results(self, results: list[dict]) -> str:
        """Format search results as a Markdown string for display."""
        if not results:
            return "_No web results found._"
        lines = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "No title")
            href = r.get("href", "")
            body = r.get("body", "")
            lines.append(f"**{i}. [{title}]({href})**\n{body}\n")
        return "\n".join(lines)
