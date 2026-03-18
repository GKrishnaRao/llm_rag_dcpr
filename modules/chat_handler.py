"""Manages chat conversation history for the Streamlit session."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")
    )
    metadata: dict = field(default_factory=dict)


class ChatHandler:
    """Stores and manages the current conversation."""

    def __init__(self) -> None:
        self._messages: list[ChatMessage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> ChatMessage:
        msg = ChatMessage(role="user", content=content)
        self._messages.append(msg)
        return msg

    def add_assistant_message(self, content: str, metadata: dict | None = None) -> ChatMessage:
        msg = ChatMessage(role="assistant", content=content, metadata=metadata or {})
        self._messages.append(msg)
        return msg

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def to_llm_history(self) -> list[dict]:
        """
        Return history in the format expected by the Claude messages API,
        excluding the most recent user turn (which is added by the agent).
        """
        history = []
        for msg in self._messages:
            if msg.role in ("user", "assistant"):
                history.append({"role": msg.role, "content": msg.content})
        # Drop the last user message — the agent adds its own enriched version
        if history and history[-1]["role"] == "user":
            history = history[:-1]
        return history

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
