"""Tests for PDF upload: the /documents endpoints and per-file ingestion.

No API key or network needed: the endpoint tests stub out ingest_file, and the
ingestion tests run against a real ChromaDB in a temp dir with fake embeddings.
"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import api
import ingest

PDF_BYTES = b"%PDF-1.7\n fake body"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", d)
    return d


@pytest.fixture
def client():
    return TestClient(api.app)


def _upload(client, name="acme-10k.pdf", body=PDF_BYTES):
    return client.post("/documents", files={"file": (name, body, "application/pdf")})


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
# POST /documents
# ---------------------------------------------------------------------------

def test_upload_ingests_under_the_original_name_and_saves_to_data(
    client, data_dir, monkeypatch
):
    seen = {}

    def fake_ingest(path):
        seen["name"] = path.name
        seen["bytes"] = path.read_bytes()
        return 42

    monkeypatch.setattr(ingest, "ingest_file", fake_ingest)
    resp = _upload(client)

    assert resp.status_code == 201
    assert resp.json() == {"source_file": "acme-10k.pdf", "chunks": 42}
    # The name ingest sees is the one citations will show
    assert seen == {"name": "acme-10k.pdf", "bytes": PDF_BYTES}
    assert (data_dir / "acme-10k.pdf").read_bytes() == PDF_BYTES
    # Staging dir is cleaned up
    assert [p.name for p in data_dir.iterdir()] == ["acme-10k.pdf"]


def test_reupload_replaces_the_saved_file(client, data_dir, monkeypatch):
    monkeypatch.setattr(ingest, "ingest_file", lambda path: 1)
    _upload(client, body=PDF_BYTES + b" v1")
    assert _upload(client, body=PDF_BYTES + b" v2").status_code == 201
    assert (data_dir / "acme-10k.pdf").read_bytes() == PDF_BYTES + b" v2"


def test_upload_rejects_non_pdf_content(client, data_dir, monkeypatch):
    monkeypatch.setattr(ingest, "ingest_file", lambda path: pytest.fail("ingested"))
    resp = _upload(client, body=b"MZ\x90\x00 not a pdf")
    assert resp.status_code == 415
    assert not data_dir.exists() or not any(data_dir.iterdir())


def test_upload_rejects_wrong_extension(client, data_dir):
    assert _upload(client, name="report.docx").status_code == 415


def test_upload_rejects_oversized_files(client, data_dir, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 16)
    assert _upload(client, body=PDF_BYTES + b"x" * 32).status_code == 413


def test_failed_ingestion_leaves_nothing_in_data(client, data_dir, monkeypatch):
    def no_text(path):
        raise ValueError(f"{path.name} has no extractable text")

    monkeypatch.setattr(ingest, "ingest_file", no_text)
    resp = _upload(client)

    assert resp.status_code == 422
    assert "no extractable text" in resp.json()["detail"]
    assert list(data_dir.iterdir()) == []


def test_list_documents(client, monkeypatch):
    monkeypatch.setattr(ingest, "list_documents", lambda: {"a.pdf": 3, "b.pdf": 7})
    resp = client.get("/documents")
    assert resp.json() == {
        "documents": [
            {"source_file": "a.pdf", "chunks": 3},
            {"source_file": "b.pdf", "chunks": 7},
        ]
    }


# ---------------------------------------------------------------------------
# ingest_file against a real (temp) ChromaDB
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(
        ingest, "OpenAIEmbeddings", lambda model: DeterministicFakeEmbedding(size=32)
    )


def _fake_pdf(monkeypatch, pages):
    def load(path):
        return [
            Document(page_content=text, metadata={"source_file": path.name, "page": i})
            for i, text in enumerate(pages)
        ]

    monkeypatch.setattr(ingest, "load_pdf", load)


def test_reingesting_a_file_replaces_its_chunks(store, tmp_path, monkeypatch):
    _fake_pdf(monkeypatch, ["page one text", "page two text"])
    ingest.ingest_file(tmp_path / "acme.pdf")
    ingest.ingest_file(tmp_path / "other.pdf")
    assert ingest.list_documents() == {"acme.pdf": 2, "other.pdf": 2}

    _fake_pdf(monkeypatch, ["revised filing, one page"])
    assert ingest.ingest_file(tmp_path / "acme.pdf") == 1
    # acme replaced, not duplicated; other untouched
    assert ingest.list_documents() == {"acme.pdf": 1, "other.pdf": 2}


def test_ingest_rejects_pdfs_without_text(store, tmp_path, monkeypatch):
    _fake_pdf(monkeypatch, ["", "   \n"])
    with pytest.raises(ValueError, match="no extractable text"):
        ingest.ingest_file(tmp_path / "scan.pdf")


def test_list_documents_is_empty_before_any_ingest(store):
    assert ingest.list_documents() == {}
