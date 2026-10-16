"""Tests for PDF upload and per-user isolation.

Covers the /documents and /query endpoints, per-file ingestion, and retrieval
filtering. No API key or network needed: the endpoint tests stub out ingestion
and querying, and the store tests run against a real ChromaDB in a temp dir
with fake embeddings and a fake reranker.
"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import api
import ingest
import rag

PDF_BYTES = b"%PDF-1.7\n fake body"
ALICE = "0f8b1c2e-alice"
BOB = "7d3a9e41-bob"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", d)
    monkeypatch.setattr(ingest, "USERS_DIR", d / "users")
    return d


@pytest.fixture
def client():
    return TestClient(api.app)


def _upload(client, name="acme-10k.pdf", body=PDF_BYTES, owner=ALICE):
    return client.post(
        "/documents",
        files={"file": (name, body, "application/pdf")},
        headers={"X-User-Id": owner},
    )


# ---------------------------------------------------------------------------
# Filename sanitising
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("acme-10k.pdf", "acme-10k.pdf"),
        ("../../etc/evil.pdf", "evil.pdf"),
        ("C:\\Users\\me\\acme 10-K (2024).PDF", "acme_10-K__2024_.PDF"),
        (".hidden.pdf", "hidden.pdf"),
    ],
)
def test_safe_pdf_name_strips_paths_and_odd_characters(raw, expected):
    assert api.safe_pdf_name(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "notes.txt", ".pdf", "../.pdf"])
def test_safe_pdf_name_rejects_non_pdf_names(raw):
    with pytest.raises(HTTPException) as exc:
        api.safe_pdf_name(raw)
    assert exc.value.status_code == 415


# ---------------------------------------------------------------------------
# User id header
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("header", [None, "", "../../etc", "a/b", "public", "x" * 65])
def test_requests_without_a_valid_user_id_are_rejected(client, header):
    headers = {} if header is None else {"X-User-Id": header}
    assert client.get("/documents", headers=headers).status_code == 401
    assert (
        client.post("/query", json={"question": "q"}, headers=headers).status_code == 401
    )


def test_query_is_scoped_to_the_caller(client, monkeypatch):
    seen = {}

    def fake_query(question, owner):
        seen["owner"] = owner
        return rag.RAGResponse(answer="ok")

    monkeypatch.setattr(rag, "query", fake_query)
    resp = client.post("/query", json={"question": "revenue?"}, headers={"X-User-Id": BOB})
    assert resp.status_code == 200
    assert seen == {"owner": BOB}


# ---------------------------------------------------------------------------
# POST /documents
# ---------------------------------------------------------------------------

def test_upload_ingests_under_the_original_name_into_the_users_dir(
    client, data_dir, monkeypatch
):
    seen = {}

    def fake_ingest(path, owner):
        seen.update(name=path.name, owner=owner, bytes=path.read_bytes())
        return 42

    monkeypatch.setattr(ingest, "ingest_file", fake_ingest)
    resp = _upload(client)

    assert resp.status_code == 201
    assert resp.json() == {"source_file": "acme-10k.pdf", "chunks": 42, "figures": 0,
                           "shared": False}
    # The name ingest sees is the one citations will show
    assert seen == {"name": "acme-10k.pdf", "owner": ALICE, "bytes": PDF_BYTES}
    user_dir = data_dir / "users" / ALICE
    assert (user_dir / "acme-10k.pdf").read_bytes() == PDF_BYTES
    # Staging dir is cleaned up, and nothing lands in the public data/ root
    assert [p.name for p in user_dir.iterdir()] == ["acme-10k.pdf"]
    assert not list(data_dir.glob("*.pdf"))


def test_two_users_can_upload_the_same_filename(client, data_dir, monkeypatch):
    monkeypatch.setattr(ingest, "ingest_file", lambda path, owner: 1)
    _upload(client, body=PDF_BYTES + b" alice", owner=ALICE)
    _upload(client, body=PDF_BYTES + b" bob", owner=BOB)
    users = data_dir / "users"
    assert (users / ALICE / "acme-10k.pdf").read_bytes().endswith(b" alice")
    assert (users / BOB / "acme-10k.pdf").read_bytes().endswith(b" bob")


def test_reupload_replaces_the_saved_file(client, data_dir, monkeypatch):
    monkeypatch.setattr(ingest, "ingest_file", lambda path, owner: 1)
    _upload(client, body=PDF_BYTES + b" v1")
    assert _upload(client, body=PDF_BYTES + b" v2").status_code == 201
    assert (data_dir / "users" / ALICE / "acme-10k.pdf").read_bytes() == PDF_BYTES + b" v2"


def test_upload_rejects_non_pdf_content(client, data_dir, monkeypatch):
    monkeypatch.setattr(ingest, "ingest_file", lambda *a: pytest.fail("ingested"))
    resp = _upload(client, body=b"MZ\x90\x00 not a pdf")
    assert resp.status_code == 415
    assert not data_dir.exists()


def test_upload_rejects_wrong_extension(client, data_dir):
    assert _upload(client, name="report.docx").status_code == 415


def test_upload_rejects_oversized_files(client, data_dir, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 16)
    assert _upload(client, body=PDF_BYTES + b"x" * 32).status_code == 413


def test_failed_ingestion_leaves_nothing_behind(client, data_dir, monkeypatch):
    def no_text(path, owner):
        raise ValueError(f"{path.name} has no extractable text")

    monkeypatch.setattr(ingest, "ingest_file", no_text)
    resp = _upload(client)

    assert resp.status_code == 422
    assert "no extractable text" in resp.json()["detail"]
    assert list((data_dir / "users" / ALICE).iterdir()) == []


def test_list_documents_shows_own_files_and_shared_samples(client, monkeypatch):
    stores = {ALICE: {"mine.pdf": 3}, BOB: {"bobs.pdf": 9}, "public": {"msft.pdf": 7}}
    monkeypatch.setattr(ingest, "list_documents", lambda owner: stores.get(owner, {}))
    resp = client.get("/documents", headers={"X-User-Id": ALICE})
    assert resp.json() == {
        "documents": [
            {"source_file": "mine.pdf", "chunks": 3, "figures": 0, "shared": False},
            {"source_file": "msft.pdf", "chunks": 7, "figures": 0, "shared": True},
        ]
    }


# ---------------------------------------------------------------------------
# Ingestion and retrieval against a real (temp) ChromaDB
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(
        ingest, "OpenAIEmbeddings", lambda model: DeterministicFakeEmbedding(size=32)
    )


def _fake_pdf(monkeypatch, pages):
    def load(path, owner):
        return [
            Document(
                page_content=text,
                metadata={"source_file": path.name, "owner": owner, "page": i},
            )
            for i, text in enumerate(pages)
        ]

    monkeypatch.setattr(ingest, "load_pdf", load)


def test_reingesting_a_file_replaces_only_that_owners_chunks(store, tmp_path, monkeypatch):
    _fake_pdf(monkeypatch, ["page one text", "page two text"])
    ingest.ingest_file(tmp_path / "acme.pdf", ALICE)
    ingest.ingest_file(tmp_path / "acme.pdf", BOB)
    ingest.ingest_file(tmp_path / "other.pdf", ALICE)

    _fake_pdf(monkeypatch, ["revised filing, one page"])
    assert ingest.ingest_file(tmp_path / "acme.pdf", ALICE) == 1
    # Alice's acme replaced, not duplicated; her other file and Bob's acme untouched
    assert ingest.list_documents(ALICE) == {"acme.pdf": 1, "other.pdf": 2}
    assert ingest.list_documents(BOB) == {"acme.pdf": 2}


def test_ingest_rejects_pdfs_without_text(store, tmp_path, monkeypatch):
    _fake_pdf(monkeypatch, ["", "   \n"])
    with pytest.raises(ValueError, match="no extractable text"):
        ingest.ingest_file(tmp_path / "scan.pdf", ALICE)


def test_list_documents_is_empty_before_any_ingest(store):
    assert ingest.list_documents(ALICE) == {}


class _FakeReranker:
    def predict(self, pairs):
        return [0.0] * len(pairs)


@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1")
def test_retrieval_never_returns_another_users_chunks(store, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_get_reranker", lambda: _FakeReranker())
    # Keep every candidate: fake embeddings give arbitrary relevance scores
    monkeypatch.setattr(rag, "SCORE_THRESHOLD", float("-inf"))
    monkeypatch.setattr(rag, "TOP_K", 100)
    _fake_pdf(monkeypatch, ["revenue was 100", "net income was 20"])
    ingest.ingest_file(tmp_path / "sample.pdf", ingest.PUBLIC_OWNER)
    ingest.ingest_file(tmp_path / "alice.pdf", ALICE)
    ingest.ingest_file(tmp_path / "bob.pdf", BOB)
    vectorstore = ingest.open_vectorstore()

    owners = {d.metadata["owner"] for d in rag.retrieve_and_rerank("revenue", vectorstore, ALICE)}
    assert owners == {ALICE, ingest.PUBLIC_OWNER}

    # No owner (CLI and evaluate.py only) searches everything
    owners = {d.metadata["owner"] for d in rag.retrieve_and_rerank("revenue", vectorstore)}
    assert owners == {ALICE, BOB, ingest.PUBLIC_OWNER}


@pytest.fixture
def keyword_only(store, monkeypatch):
    """Fake embeddings rank arbitrarily, so drop every semantic hit and test BM25 alone."""
    monkeypatch.setattr(rag, "_get_reranker", lambda: _FakeReranker())
    monkeypatch.setattr(rag, "SCORE_THRESHOLD", float("inf"))
    monkeypatch.setattr(rag, "TOP_K", 100)


@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1")
def test_keyword_search_never_returns_another_users_chunks(keyword_only, tmp_path, monkeypatch):
    _fake_pdf(monkeypatch, ["Short-term unearned revenue 57,582"])
    ingest.ingest_file(tmp_path / "sample.pdf", ingest.PUBLIC_OWNER)
    ingest.ingest_file(tmp_path / "alice.pdf", ALICE)
    ingest.ingest_file(tmp_path / "bob.pdf", BOB)
    vectorstore = ingest.open_vectorstore()

    for asker, visible in [(ALICE, {ALICE, ingest.PUBLIC_OWNER}), (BOB, {BOB, ingest.PUBLIC_OWNER})]:
        found = rag.retrieve_and_rerank("unearned revenue 57,582", vectorstore, asker)
        assert {d.metadata["owner"] for d in found} == visible


@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1")
def test_keyword_hits_carry_their_provenance(keyword_only, tmp_path, monkeypatch):
    _fake_pdf(monkeypatch, ["Total revenue 245,122", "Goodwill 119,220"])
    ingest.ingest_file(tmp_path / "acme.pdf", ALICE)
    vectorstore = ingest.open_vectorstore()

    [doc] = rag.retrieve_and_rerank("Where is 245,122 reported?", vectorstore, ALICE)
    assert (doc.metadata["source_file"], doc.metadata["page"]) == ("acme.pdf", 0)
    trace = doc.metadata["retrieval"]
    assert trace["matched_by"] == ["keyword"]
    assert trace["keyword_rank"] == 1
    assert trace["matched_terms"] == ["245122"]
    assert trace["exact_value_match"] is True
    assert trace["chunk_id"] == doc.id

    [source] = rag.build_sources("Where is 245,122 reported?", [doc])
    assert source["retrieval"] is trace
    # The stored chunk isn't touched: the trace lives on a copy
    stored = vectorstore.get(ids=[doc.id], include=["metadatas"])["metadatas"][0]
    assert "retrieval" not in stored


@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1")
def test_keyword_search_forgets_replaced_filings(keyword_only, tmp_path, monkeypatch):
    vectorstore = ingest.open_vectorstore()
    _fake_pdf(monkeypatch, ["Goodwill 119,220"])
    ingest.ingest_file(tmp_path / "acme.pdf", ALICE)
    assert rag.retrieve_and_rerank("goodwill", vectorstore, ALICE)

    _fake_pdf(monkeypatch, ["Revised: total revenue 245,122"])
    ingest.ingest_file(tmp_path / "acme.pdf", ALICE)
    assert rag.retrieve_and_rerank("goodwill", vectorstore, ALICE) == []
    [doc] = rag.retrieve_and_rerank("revenue", vectorstore, ALICE)
    assert doc.page_content == "Revised: total revenue 245,122"


@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1")
def test_hybrid_search_can_be_switched_off(keyword_only, tmp_path, monkeypatch):
    _fake_pdf(monkeypatch, ["Goodwill 119,220"])
    ingest.ingest_file(tmp_path / "acme.pdf", ALICE)
    vectorstore = ingest.open_vectorstore()
    # Semantic hits are all filtered out here, so without keyword search nothing is left
    assert rag.retrieve_and_rerank("goodwill", vectorstore, ALICE, keyword=False) == []
    monkeypatch.setattr(rag, "HYBRID_SEARCH", False)
    assert rag.retrieve_and_rerank("goodwill", vectorstore, ALICE) == []


def test_main_ingests_public_samples_and_each_users_uploads(
    store, data_dir, monkeypatch
):
    (data_dir / "users" / ALICE).mkdir(parents=True)
    (data_dir / "sample.pdf").write_bytes(PDF_BYTES)
    (data_dir / "users" / ALICE / "mine.pdf").write_bytes(PDF_BYTES)
    calls = []
    monkeypatch.setattr(ingest, "ingest_file", lambda path, owner: calls.append((path.name, owner)))

    ingest.main()

    assert calls == [("sample.pdf", ingest.PUBLIC_OWNER), ("mine.pdf", ALICE)]
