"""
app.py — Streamlit chat interface for the Financial Report Q&A Engine, with
sign-in and a sidebar for uploading and indexing your own 10-K PDFs.

Usage:
    streamlit run financial_rag/app.py

Requires the FastAPI server to be running:
    uvicorn financial_rag.api:app --reload --port 8000

Login uses Streamlit's built-in OIDC support (st.login) against Amazon Cognito,
configured in .streamlit/secrets.toml (see .streamlit/secrets.toml.example).
Every API call carries the user's Cognito ID token, which the API verifies.
For local development without Cognito, set AUTH_DISABLED=1 to run as a single
"local" user; start.py does this automatically when no login is configured.
"""

import base64
import json
import os
import time

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


class StreamFailed(Exception):
    """A streamed answer ended with an error event (POST /query/stream)."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _ask(payload: dict) -> dict:
    """
    Ask the API, showing each step and the answer as it's written; returns the
    checked answer as POST /query would. What was streamed is only a preview: the
    final answer (after the check against the filing, and any correction) replaces it.
    """
    status = st.status("Reading the question …")
    preview = st.empty()
    streamed = ""
    with httpx.stream("POST", f"{API_BASE_URL}/query/stream", json=payload, headers=HEADERS,
                      timeout=120.0) as response:
        if response.status_code >= 400:
            response.read()             # so _error_detail can read the JSON detail
            response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "step":
                status.update(label=f"{event['step']} …")
            elif event["type"] == "token":
                streamed += event["text"]
                preview.markdown(_md(streamed) + " ▌")
            elif event["type"] == "retry":
                streamed = ""
                preview.empty()
                status.update(label="Correcting a figure the check flagged …")
            elif event["type"] == "done":
                preview.empty()
                status.update(label="Answered", state="complete")
                return event["response"]
            elif event["type"] == "error":
                preview.empty()
                status.update(label="Couldn't answer", state="error")
                raise StreamFailed(event["status"], event["detail"])
    raise StreamFailed(502, "The answer stream ended early.")


def _error_detail(exc: Exception) -> str:
    if isinstance(exc, StreamFailed):
        return exc.detail if exc.status in (401, 429) else f"API error ({exc.status}): {exc.detail}"
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail", str(exc))
        except ValueError:
            detail = exc.response.text
        if exc.response.status_code in (401, 429):
            # Sign-in expired, or a usage limit: the detail is written for the user
            return detail
        return f"API error ({exc.response.status_code}): {detail}"
    if isinstance(exc, httpx.ConnectError):
        return "Could not connect to the API server."
    return f"Unexpected error: {exc}"


def _token_expired(token: str, margin: int = 60) -> bool:
    """Whether a JWT's exp claim has passed (or will within `margin` seconds).

    Read without checking the signature: this only decides whether to ask the
    user to sign in again. The API verifies the token properly.
    """
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return claims["exp"] - margin < time.time()
    except (IndexError, KeyError, TypeError, ValueError):
        return True


def _md(text: str) -> str:
    """Escape $ so Streamlit doesn't render the text between two of them as LaTeX.

    Filings and answers are full of dollar amounts; unescaped, a passage like
    "$6.8 billion, $5.2 billion" turns into a garbled math span.
    """
    return text.replace("$", "\\$")


ALL_FILINGS = "All filings"


def _doc_line(doc: dict) -> str:
    """'Microsoft Corporation · 155 chunks · 59 figures'"""
    stats = _doc_stats(doc)
    return f"{doc['company']} · {stats}" if doc.get("company") else stats


def _doc_stats(doc: dict) -> str:
    """'155 chunks · 66 figures': figures feed the calculation engine."""
    stats = f"{doc['chunks']} chunks"
    if doc.get("figures"):
        stats += f" · {doc['figures']} figures"
    return stats


def _found_by(retrieval: dict | None) -> str:
    """One line on how a cited chunk was retrieved, e.g. "semantic + keyword match"."""
    if not retrieval:
        return ""
    line = " + ".join(retrieval.get("matched_by", [])) + " match"
    if retrieval.get("matched_terms"):
        line += " · terms: " + ", ".join(retrieval["matched_terms"])
    if retrieval.get("llm_rank"):
        line += f" · reranked #{retrieval['llm_rank']} by LLM"
    return line


def _usage_line(counts: dict) -> str:
    """'Questions: 12 of 50 · Uploads: 1 of 3 this month'"""
    def part(label: str, used: int, limit: int) -> str:
        return f"{label}: {used} of {limit}" if limit else f"{label}: {used}"

    return (part("Questions", counts["questions"], counts["question_limit"]) + " · "
            + part("Uploads", counts["uploads"], counts["upload_limit"]) + " this month")


FETCH_TIMEOUT = 15 * 60     # seconds: a long 10-K can take several minutes to index


def _fetch_filing(ticker: str, fiscal_year: int | None = None) -> None:
    """Ask the API to fetch a 10-K by ticker (the latest, or one fiscal year's), then
    follow the job until it ends."""
    try:
        response = httpx.post(f"{API_BASE_URL}/filings",
                              json={"ticker": ticker, "fiscal_year": fiscal_year},
                              headers=HEADERS, timeout=30.0)
        response.raise_for_status()
        job = response.json()
    except Exception as exc:
        st.error(f"{ticker}: {_error_detail(exc)}")
        return
    which = f"{ticker.upper()}'s FY{fiscal_year} 10-K" if fiscal_year else \
        f"{job['subject']}'s latest 10-K"
    with st.status(f"Fetching {which} …", expanded=True) as status:
        shown = ""
        deadline = time.time() + FETCH_TIMEOUT
        while job["state"] not in ("done", "failed") and time.time() < deadline:
            if job.get("step") and job["step"] != shown:
                shown = job["step"]
                status.write(shown)
            time.sleep(2)
            try:
                response = httpx.get(f"{API_BASE_URL}/jobs/{job['job_id']}",
                                     headers=HEADERS, timeout=10.0)
                response.raise_for_status()
                job = response.json()
            except Exception as exc:
                status.update(label=f"{ticker}: {_error_detail(exc)}", state="error")
                return
        if job["state"] == "done":
            doc = job["document"]
            status.update(label=f"Indexed {doc['source_file']} ({_doc_line(doc)})",
                          state="complete", expanded=False)
        elif job["state"] == "failed":
            status.update(label=f"{job['subject']}: {job['error']}", state="error")
        else:
            status.update(label=f"{job['subject']} is still being indexed; it will appear "
                                "under your filings when it's done.", state="running")


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
            "Sample filings are ready to query as soon as you sign in."
        )
        st.button("Sign in or create an account", on_click=st.login, type="primary")
        st.stop()
    id_token = getattr(st.user, "tokens", {}).get("id")
    if not id_token:
        st.error('Login is missing `expose_tokens = "id"` in the [auth] section of '
                 ".streamlit/secrets.toml, so the API can't verify who you are.")
        st.stop()
    if _token_expired(id_token):
        # Cognito ID tokens last an hour; signing in again is one click while
        # Cognito's own session lasts.
        st.info("Your sign-in has expired.")
        st.button("Sign in again", on_click=st.logout, type="primary")
        st.stop()
    OWNER = st.user.get("sub")
    USER_LABEL = st.user.get("email", "Signed in")
    # The API verifies this token and takes the user from it (auth.py)
    HEADERS = {"Authorization": f"Bearer {id_token}"}
elif os.getenv("AUTH_DISABLED") == "1":
    OWNER = "local"
    USER_LABEL = "Local mode (login disabled)"
    # Local mode: the API trusts this header only when it runs with AUTH_DISABLED=1 too
    HEADERS = {"X-User-Id": OWNER}
else:
    st.error(
        "Login isn't configured. Add an [auth] section to .streamlit/secrets.toml, "
        "or set AUTH_DISABLED=1 to run as a single local user."
    )
    st.stop()

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
                    st.success(f"Indexed {info['source_file']} ({_doc_stats(info)})")
                except Exception as exc:
                    st.error(f"{upload.name}: {_error_detail(exc)}")

    with st.form("fetch", clear_on_submit=True):
        ticker = st.text_input("Or fetch a company's 10-K", placeholder="Ticker, e.g. AAPL",
                               max_chars=12)
        year = st.number_input(
            "Fiscal year", value=None, min_value=1994, max_value=2100, step=1,
            placeholder="Latest",
            help="Blank for the latest 10-K. Fetch earlier years too, and questions about "
                 "the company draw on all of them: a longer history, restatements included.")
        fetch = st.form_submit_button("Fetch from SEC EDGAR")
    if fetch and ticker.strip():
        _fetch_filing(ticker.strip(), int(year) if year else None)

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
            st.markdown(f"- **{doc['source_file']}** — {_doc_line(doc)}")
    else:
        st.info("You haven't uploaded any filings yet.")
    if shared:
        st.caption("Sample filings (available to everyone)")
        for doc in shared:
            st.markdown(f"- **{doc['source_file']}** — {_doc_line(doc)}")

    # Which filing questions are about. A user's own file shadows a sample of
    # the same name, as it does in the API.
    companies = {}
    for doc in shared + own:
        companies[doc["source_file"]] = doc.get("company")
    choices = [ALL_FILINGS, *sorted(companies)]
    if st.session_state.get("ask_about") not in choices:
        st.session_state.ask_about = ALL_FILINGS
    st.selectbox(
        "Ask about",
        choices,
        key="ask_about",
        format_func=lambda name: name if name == ALL_FILINGS or not companies.get(name)
        else f"{companies[name]} ({name})",
        help="Answer from one filing only. With all filings, naming a company in the "
             "question narrows the search to its filing.",
    )

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("interpreted"):
            st.caption(_md(f"Read as: {msg['interpreted']}"))
        st.markdown(_md(msg["content"]))
        if msg.get("filing"):
            st.caption(f"Answered from {msg['filing']} only.")
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"], expanded=False)

# Chat input
if question := st.chat_input("Ask a question about the financial reports …"):
    # Earlier turns, so the API can read a follow-up ("and in 2023?") in context
    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]
    ]
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(_md(question))

    # Call the API
    with st.chat_message("assistant"):
        try:
            ask_about = st.session_state.get("ask_about", ALL_FILINGS)
            data = _ask({
                "question": question,
                "source_file": None if ask_about == ALL_FILINGS else ask_about,
                "history": history,
            })
            answer = data["answer"]
            sources = data.get("sources", [])
            # "a.htm" or, for a comparison, "a.htm and b.htm"
            filing = data.get("filing") or " and ".join(data.get("filings") or []) or None
            interpreted = data.get("question") or question
        except httpx.ConnectError:
            answer = (
                "Could not connect to the API server. "
                "Make sure `uvicorn financial_rag.api:app --reload --port 8000` is running."
            )
            sources = []
            filing = None
            interpreted = question
        except (httpx.HTTPStatusError, StreamFailed) as exc:
            answer = _error_detail(exc)
            sources = []
            filing = None
            interpreted = question
        except Exception as exc:
            answer = f"Unexpected error: {exc}"
            sources = []
            filing = None
            interpreted = question

        if interpreted != question:
            st.caption(_md(f"Read as: {interpreted}"))
        st.markdown(_md(answer))
        if filing:
            st.caption(f"Answered from {filing} only.")

        if sources:
            _render_sources(sources, expanded=True)

    # Persist assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "filing": filing,
         "interpreted": interpreted if interpreted != question else None}
    )

# Last, so the counts include the question just asked
with st.sidebar:
    try:
        response = httpx.get(f"{API_BASE_URL}/usage", headers=HEADERS, timeout=10.0)
        response.raise_for_status()
        st.caption(_usage_line(response.json()))
    except Exception:
        pass   # the counts are informational; the API still enforces the limits
