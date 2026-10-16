"""
app.py — Streamlit chat interface for the Financial Report Q&A Engine, with
sign-in and a sidebar for uploading and indexing your own 10-K PDFs.

Usage:
    streamlit run app.py

Requires the FastAPI server to be running:
    uvicorn api:app --reload --port 8000

Login uses Streamlit's built-in OIDC support (st.login) against Amazon Cognito,
configured in .streamlit/secrets.toml (see .streamlit/secrets.toml.example).
For local development without Cognito, set AUTH_DISABLED=1 to run as a single
"local" user; start.py does this automatically when no login is configured.
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


def _auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        # Raised when there is no secrets.toml at all
        return False


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


def _found_by(retrieval: dict | None) -> str:
    """One line on how a cited chunk was retrieved, e.g. "semantic + keyword match"."""
    if not retrieval:
        return ""
    line = " + ".join(retrieval.get("matched_by", [])) + " match"
    if retrieval.get("matched_terms"):
        line += " · terms: " + ", ".join(retrieval["matched_terms"])
    return line


def _render_sources(sources: list[dict], expanded: bool) -> None:
    with st.expander("Source citations", expanded=expanded):
        for src in sources:
            st.markdown(
                f"**{src['source_file']}** — Page {src['page']}\n\n"
                f"> {_md(src['excerpt'])}"
            )
            if found := _found_by(src.get("retrieval")):
                st.caption(_md(found))


# Who is using the app. Fail closed: with neither login nor the explicit
# local-mode switch configured, nobody gets in.
if _auth_configured():
    if not st.user.is_logged_in:
        st.markdown(
            "Upload 10-K filings and ask questions about them. Every answer cites "
            "the page it came from, and figures that aren't in the filing are never "
            "guessed.\n\n"
            "A sample filing (Microsoft FY2024 10-K) is ready to query as soon as "
            "you sign in."
        )
        st.button("Sign in or create an account", on_click=st.login, type="primary")
        st.stop()
    OWNER = st.user.get("sub")
    USER_LABEL = st.user.get("email", "Signed in")
elif os.getenv("AUTH_DISABLED") == "1":
    OWNER = "local"
    USER_LABEL = "Local mode (login disabled)"
else:
    st.error(
        "Login isn't configured. Add an [auth] section to .streamlit/secrets.toml, "
        "or set AUTH_DISABLED=1 to run as a single local user."
    )
    st.stop()

# The API trusts this header; it is only reachable from this server.
HEADERS = {"X-User-Id": OWNER}

# Chat history belongs to one user: reset it if someone else signs in
# in the same browser session.
if st.session_state.get("owner") != OWNER:
    st.session_state.owner = OWNER
    st.session_state.messages = []

# Sidebar: account, upload filings, and show what is already indexed
with st.sidebar:
    st.caption(USER_LABEL)
    if _auth_configured():
        st.button("Sign out", on_click=st.logout)

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
                        headers=HEADERS,
                        # Embedding a full 10-K takes a while
                        timeout=300.0,
                    )
                    response.raise_for_status()
                    info = response.json()
                    st.success(f"Indexed {info['source_file']} ({info['chunks']} chunks)")
                except Exception as exc:
                    st.error(f"{upload.name}: {_error_detail(exc)}")

    try:
        response = httpx.get(f"{API_BASE_URL}/documents", headers=HEADERS, timeout=10.0)
        response.raise_for_status()
        documents = response.json()["documents"]
    except Exception as exc:
        documents = []
        st.warning(_error_detail(exc))
    own = [d for d in documents if not d.get("shared")]
    shared = [d for d in documents if d.get("shared")]
    st.caption("Your filings")
    if own:
        for doc in own:
            st.markdown(f"- **{doc['source_file']}** — {doc['chunks']} chunks")
    else:
        st.info("You haven't uploaded any filings yet.")
    if shared:
        st.caption("Sample filings (available to everyone)")
        for doc in shared:
            st.markdown(f"- **{doc['source_file']}** — {doc['chunks']} chunks")

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
                    headers=HEADERS,
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
