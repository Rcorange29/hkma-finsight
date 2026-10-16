"""Tests for verifying the Cognito ID token (auth.py) and how the API applies it.

Tokens are real RS256 JWTs signed with a key generated here; the JWKS fetch is
replaced by that key, so no network is needed.
"""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from financial_rag import api, auth, ingest

ISSUER = "https://cognito-idp.ap-southeast-2.amazonaws.com/ap-southeast-2_TestPool"
CLIENT = "test-client-id"
ALICE = "0f8b1c2e-1111-4a5b-9c3d-alice0000000"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeJWKS:
    """Stands in for PyJWKClient: always returns this test's public key."""

    def get_signing_key_from_jwt(self, token):
        return type("SigningKey", (), {"key": KEY.public_key()})()


@pytest.fixture(autouse=True)
def cognito(monkeypatch):
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    monkeypatch.setenv("COGNITO_ISSUER", ISSUER)
    monkeypatch.setenv("COGNITO_CLIENT_ID", CLIENT)
    monkeypatch.setattr(auth, "_jwks_client", lambda iss: FakeJWKS())


def token(key=KEY, **overrides) -> str:
    now = int(time.time())
    claims = {"sub": ALICE, "iss": ISSUER, "aud": CLIENT, "token_use": "id",
              "iat": now, "exp": now + 3600, "email": "alice@example.com"}
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test"})


def bearer(t: str) -> str:
    return f"Bearer {t}"


# ---------------------------------------------------------------------------
# auth.verify
# ---------------------------------------------------------------------------

def test_a_valid_id_token_gives_its_claims():
    assert auth.verify(bearer(token()))["sub"] == ALICE


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: None, "Missing sign-in token"),
        (lambda: "Basic abc", "Missing sign-in token"),
        (lambda: "Bearer ", "Missing sign-in token"),
        (lambda: bearer("not.a.jwt"), "Invalid sign-in token"),
        (lambda: bearer(token(exp=int(time.time()) - 120)), "expired"),
        (lambda: bearer(token(aud="another-app")), "Invalid sign-in token"),
        (lambda: bearer(token(iss="https://evil.example.com")), "Invalid sign-in token"),
        (lambda: bearer(token(token_use="access")), "Invalid sign-in token"),
        (lambda: bearer(token(key=OTHER_KEY)), "Invalid sign-in token"),   # forged signature
        (lambda: bearer(token(exp=None)), "Invalid sign-in token"),        # no expiry
    ],
)
def test_bad_tokens_are_rejected(make, message):
    with pytest.raises(auth.AuthError) as exc:
        auth.verify(make())
    assert message in str(exc.value)
    assert exc.value.status == 401


def test_unsigned_tokens_are_rejected():
    unsigned = jwt.encode({"sub": ALICE, "iss": ISSUER, "aud": CLIENT, "token_use": "id",
                           "iat": int(time.time()), "exp": int(time.time()) + 60},
                          key=None, algorithm="none")
    with pytest.raises(auth.AuthError):
        auth.verify(bearer(unsigned))


def test_small_clock_skew_is_tolerated():
    assert auth.verify(bearer(token(exp=int(time.time()) - 10)))["sub"] == ALICE


def test_unconfigured_server_refuses_everyone(monkeypatch):
    monkeypatch.delenv("COGNITO_ISSUER")
    with pytest.raises(auth.AuthError) as exc:
        auth.verify(bearer(token()))
    assert exc.value.status == 503


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def no_filings(monkeypatch):
    monkeypatch.setattr(ingest, "filings", lambda owners: [])


def test_api_takes_the_user_from_the_token(client, no_filings, monkeypatch):
    seen = []
    monkeypatch.setattr(ingest, "filings", lambda owners: seen.append(owners) or [])
    response = client.get("/documents", headers={"Authorization": bearer(token())})
    assert response.status_code == 200
    assert seen == [[ALICE, ingest.PUBLIC_OWNER]]


def test_x_user_id_cannot_override_the_token(client, monkeypatch):
    seen = []
    monkeypatch.setattr(ingest, "filings", lambda owners: seen.append(owners) or [])
    client.get("/documents", headers={"Authorization": bearer(token()),
                                      "X-User-Id": "7d3a9e41-bob"})
    assert seen == [[ALICE, ingest.PUBLIC_OWNER]]


def test_x_user_id_alone_is_not_enough(client, no_filings):
    response = client.get("/documents", headers={"X-User-Id": ALICE})
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing sign-in token."


def test_expired_token_gets_a_clear_401(client, no_filings):
    response = client.get("/documents", headers={
        "Authorization": bearer(token(exp=int(time.time()) - 120))})
    assert response.status_code == 401
    assert "expired" in response.json()["detail"]


def test_every_user_endpoint_needs_a_token(client):
    for method, path in [("get", "/documents"), ("get", "/financials"), ("get", "/usage"),
                         ("post", "/query"), ("post", "/documents")]:
        assert getattr(client, method)(path).status_code == 401, path
    assert client.get("/health").status_code == 200


def test_unconfigured_api_answers_503(client, monkeypatch):
    monkeypatch.delenv("COGNITO_CLIENT_ID")
    response = client.get("/documents", headers={"Authorization": bearer(token())})
    assert response.status_code == 503


def test_local_mode_trusts_the_header(client, no_filings, monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "1")
    assert client.get("/documents", headers={"X-User-Id": "local"}).status_code == 200
