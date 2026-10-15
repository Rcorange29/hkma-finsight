"""
app.py — Streamlit chat interface for the Financial Report Q&A Engine, with a
sidebar for uploading and indexing your own 10-K PDFs.

Usage:
    streamlit run app.py

Requires the FastAPI server to be running:
    uvicorn api:app --reload --port 8000
"""

import os

import httpx
import streamlit as st

# Overridden in Docker, where the API is reachable by service name, not localhost.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Financial Report Q&A",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Financial Report Q&A Engine")
st.caption("Ask questions about 10-K filings — powered by RAG + GPT-4o-mini")


def _error_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail", str(exc))
        except ValueError:
            detail = exc.response.text
        return f"API error ({exc.response.status_code}): {detail}"
    if isinstance(exc, httpx.ConnectError):
        return "Could not connect to the API server."
    return f"Unexpected error: {exc}"


def _md(text: str) -> str:
    """Escape $ so Streamlit doesn't render the text between two of them as LaTeX.

    Filings and answers are full of dollar amounts; unescaped, a passage like
    "$6.8 billion, $5.2 billion" turns into a garbled math span.
    """
    return text.replace("$", "\\$")


def _render_sources(sources: list[dict], expanded: bool) -> None:
    with st.expander("Source citations", expanded=expanded):
        for src in sources:
            st.markdown(
                f"**{src['source_file']}** — Page {src['page']}\n\n"
                f"> {_md(src['excerpt'])}"
            )


# Sidebar: upload filings and show what is already indexed
with st.sidebar:
    st.header("Documents")
    with st.form("upload", clear_on_submit=True):
        uploads = st.file_uploader(
            "Upload 10-K PDFs", type="pdf", accept_multiple_files=True
        )
        submitted = st.form_submit_button("Upload and index")
    if submitted and uploads:
        for upload in uploads:
            with st.spinner(f"Indexing {upload.name} …"):
                try:
                    response = httpx.post(
                        f"{API_BASE_URL}/documents",
                        files={"file": (upload.name, upload.getvalue(), "application/pdf")},
                        # Embedding a full 10-K takes a while
                        timeout=300.0,
                    )
                    response.raise_for_status()
                    info = response.json()
                    st.success(f"Indexed {info['source_file']} ({info['chunks']} chunks)")
                except Exception as exc:
                    st.error(f"{upload.name}: {_error_detail(exc)}")

    try:
        response = httpx.get(f"{API_BASE_URL}/documents", timeout=10.0)
        response.raise_for_status()
        documents = response.json()["documents"]
    except Exception as exc:
        documents = []
        st.warning(_error_detail(exc))
    if documents:
        st.caption("Indexed filings")
        for doc in documents:
            st.markdown(f"- **{doc['source_file']}** — {doc['chunks']} chunks")
    else:
        st.info("No filings indexed yet. Upload a PDF to get started.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(_md(msg["content"]))
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"], expanded=False)

# Chat input
if question := st.chat_input("Ask a question about the financial reports …"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(_md(question))

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and generating answer …"):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/query",
                    json={"question": question},
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])
            except httpx.ConnectError:
                answer = (
                    "Could not connect to the API server. "
                    "Make sure `uvicorn api:app --reload --port 8000` is running."
                )
                sources = []
            except httpx.HTTPStatusError as exc:
                detail = exc.response.json().get("detail", str(exc))
                answer = f"API error ({exc.response.status_code}): {detail}"
                sources = []
            except Exception as exc:
                answer = f"Unexpected error: {exc}"
                sources = []

        st.markdown(_md(answer))

        if sources:
            _render_sources(sources, expanded=True)

    # Persist assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
