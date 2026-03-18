"""
RAG Document Assistant — Streamlit application entry point.

Two-panel layout:
  Left sidebar  : file upload + list of recent documents
  Main area     : chat interface with RAG agent
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import streamlit as st

from config import Config
from modules.chat_handler import ChatHandler
from modules.document_processor import DocumentProcessor
from modules.gcs_handler import GCSHandler
from modules.rag_agent import RAGAgent
from modules.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Real AI Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Chat bubble styles */
    .user-bubble {
        background: #1e3a5f;
        color: #e8f4fd;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 8px 0 8px 15%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .assistant-bubble {
        background: #1a1a2e;
        color: #e8e8f0;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 8px 15% 8px 0;
        border-left: 3px solid #4a9eff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .timestamp {
        font-size: 0.7rem;
        color: #888;
        margin-top: 4px;
    }
    .source-tag {
        background: #0e3460;
        color: #7ec8f0;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-right: 4px;
    }
    .web-tag {
        background: #1a3320;
        color: #6ecb8a;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
    }
    .doc-item {
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4a9eff;
        background: #ffffff;
        margin-bottom: 8px;
        font-size: 0.85rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: box-shadow 0.15s ease;
    }
    .doc-item:hover {
        box-shadow: 0 3px 10px rgba(74,158,255,0.15);
    }
    .doc-item b {
        color: #1a202c;
        display: block;
        margin-bottom: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }
    .doc-meta {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.73rem;
        color: #718096;
    }
    .doc-badge {
        background: #ebf4ff;
        color: #2b6cb0;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    if "chat_handler" not in st.session_state:
        st.session_state.chat_handler = ChatHandler()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if "indexed_files" not in st.session_state:
        # Set of filenames already added to the vector store this session
        st.session_state.indexed_files: set[str] = set()
    if "gcs_handler" not in st.session_state:
        try:
            st.session_state.gcs_handler = GCSHandler()
        except Exception:
            st.session_state.gcs_handler = None
    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = RAGAgent(st.session_state.vector_store)


# ---------------------------------------------------------------------------
# Sidebar — file upload + document list
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.title("📂 Documents")
        st.markdown("---")

        # Config validation warning
        missing = Config.validate()
        if missing:
            st.warning(
                f"⚠️ Missing config: **{', '.join(missing)}**\n\n"
                "Copy `.env.example` → `.env` and fill in the values.",
                icon="⚠️",
            )

        _render_upload_section()
        st.markdown("---")
        _render_document_list()

        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_handler.clear()
            st.rerun()


def _render_upload_section() -> None:
    st.subheader("Upload Document")
    uploaded = st.file_uploader(
        "Choose a file",
        type=[ext.lstrip(".") for ext in Config.SUPPORTED_EXTENSIONS],
        help=f"Supported: {', '.join(Config.SUPPORTED_EXTENSIONS)}",
    )

    if uploaded is not None:
        col1, col2 = st.columns(2)
        with col1:
            do_upload = st.button("☁️ Upload to GCS", use_container_width=True)
        with col2:
            do_index = st.button("⚡ Index Only", use_container_width=True)

        if do_upload:
            _handle_upload_to_gcs(uploaded)
        elif do_index:
            _handle_index_only(uploaded)


def _handle_upload_to_gcs(uploaded_file) -> None:
    gcs: GCSHandler | None = st.session_state.gcs_handler
    if gcs is None or not Config.GCS_BUCKET_NAME:
        st.error("GCS is not configured. Use '⚡ Index Only' to index locally.")
        return

    with st.spinner(f"Uploading **{uploaded_file.name}** to GCS…"):
        try:
            meta = gcs.upload_file(uploaded_file, uploaded_file.name)
            st.success(f"✅ Uploaded: **{meta['filename']}**")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")
            return

    _index_file_bytes(uploaded_file.read() if uploaded_file.tell() == 0 else _reread(uploaded_file),
                      uploaded_file.name)


def _handle_index_only(uploaded_file) -> None:
    file_bytes = uploaded_file.read()
    _index_file_bytes(file_bytes, uploaded_file.name)


def _index_file_bytes(file_bytes: bytes, filename: str) -> None:
    if filename in st.session_state.indexed_files:
        st.info(f"**{filename}** is already indexed in this session.")
        return

    processor = DocumentProcessor()
    vs: VectorStore = st.session_state.vector_store

    with st.spinner(f"Indexing **{filename}**…"):
        try:
            docs = processor.process(file_bytes, filename)
            vs.add_documents(docs)
            st.session_state.indexed_files.add(filename)
            st.success(f"✅ Indexed **{filename}** ({len(docs)} chunks)")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")


def _reread(uploaded_file) -> bytes:
    uploaded_file.seek(0)
    return uploaded_file.read()


def _render_document_list() -> None:
    st.subheader("Recent Documents")

    gcs: GCSHandler | None = st.session_state.gcs_handler

    # Show GCS documents if available
    if gcs is not None and Config.GCS_BUCKET_NAME:
        try:
            docs = gcs.list_documents(limit=Config.MAX_DOCUMENTS_DISPLAY)
            if docs:
                for doc in docs:
                    ts = doc["uploaded_at"][:10]
                    ext = Path(doc["filename"]).suffix.upper().lstrip(".") or "FILE"
                    st.markdown(
                        f"""<div class="doc-item">
                            <b>📄 {doc['filename']}</b>
                            <div class="doc-meta">
                                <span class="doc-badge">{ext}</span>
                                <span>📅 {ts}</span>
                                <span>💾 {doc['size_kb']} KB</span>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No documents in GCS yet.")
        except Exception as exc:
            st.caption(f"Could not load GCS list: {exc}")
    else:
        # Show locally indexed files
        indexed = st.session_state.indexed_files
        if indexed:
            for fname in list(indexed)[:Config.MAX_DOCUMENTS_DISPLAY]:
                ext = Path(fname).suffix.upper().lstrip(".") or "FILE"
                st.markdown(
                    f"""<div class="doc-item">
                        <b>📄 {fname}</b>
                        <div class="doc-meta">
                            <span class="doc-badge">{ext}</span>
                            <span>✅ Indexed</span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No documents indexed yet. Upload a file above.")


# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------

def render_chat() -> None:
    st.title("🔍 Real AI Assistant")
    st.caption(
        "Ask questions about your uploaded documents. "
        "If no answer is found in the documents, the agent will search the web."
    )

    chat: ChatHandler = st.session_state.chat_handler
    agent: RAGAgent = st.session_state.rag_agent

    # --- Display history ------------------------------------------------
    chat_container = st.container()

    with chat_container:
        if not chat.messages:
            st.info(
                "👋 **Welcome!** Upload a document on the left, then ask me anything about it.",
                icon="ℹ️",
            )
        else:
            for msg in chat.messages:
                _render_message(msg)

    # --- Input ----------------------------------------------------------
    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([8, 1])
        with cols[0]:
            user_input = st.text_area(
                "Your question",
                placeholder="e.g. What are the main compliance requirements in section 3?",
                height=80,
                label_visibility="collapsed",
            )
        with cols[1]:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_input.strip():
        _process_question(user_input.strip(), chat, agent)
        st.rerun()


def _render_message(msg) -> None:
    if msg.role == "user":
        st.markdown(
            f'<div class="user-bubble">👤 {msg.content}'
            f'<div class="timestamp">{msg.timestamp}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Build source badges
        source_docs: list = msg.metadata.get("source_documents", [])
        used_web: bool = msg.metadata.get("used_web_search", False)

        badges = ""
        seen_sources: set[str] = set()
        for doc in source_docs:
            src = doc.metadata.get("source", "doc")
            if src not in seen_sources:
                badges += f'<span class="source-tag">📄 {Path(src).name}</span>'
                seen_sources.add(src)
        if used_web:
            badges += '<span class="web-tag">🌐 Web search</span>'

        st.markdown(
            f'<div class="assistant-bubble">🤖 {msg.content}'
            f'{"<br>" + badges if badges else ""}'
            f'<div class="timestamp">{msg.timestamp}</div></div>',
            unsafe_allow_html=True,
        )

        # Expandable source excerpts
        if source_docs:
            with st.expander("📖 View source excerpts", expanded=False):
                for i, doc in enumerate(source_docs, start=1):
                    src = doc.metadata.get("source", "unknown")
                    st.markdown(f"**[{i}] {src}**")
                    st.text(doc.page_content[:500] + ("…" if len(doc.page_content) > 500 else ""))

        # Expandable web results
        web_results: list[dict] = msg.metadata.get("web_results", [])
        if web_results:
            with st.expander("🌐 View web search results", expanded=False):
                from modules.search_handler import SearchHandler
                sh = SearchHandler()
                st.markdown(sh.format_results(web_results))


def _process_question(question: str, chat: ChatHandler, agent: RAGAgent) -> None:
    chat.add_user_message(question)

    with st.spinner("Thinking…"):
        history = chat.to_llm_history()
        try:
            result = agent.answer(question, conversation_history=history)
        except Exception as exc:
            logger.error("Agent error: %s", exc)
            result_answer = f"An error occurred while processing your question: {exc}"
            chat.add_assistant_message(result_answer)
            return

    chat.add_assistant_message(
        result.answer,
        metadata={
            "source_documents": result.source_documents,
            "web_results": result.web_results,
            "used_web_search": result.used_web_search,
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
