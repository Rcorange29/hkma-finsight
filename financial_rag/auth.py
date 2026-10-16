"""
auth.py — Who is calling the API, proved by a Cognito ID token.

The UI signs users in with Cognito and sends their ID token as
`Authorization: Bearer <token>`. The API checks the token's signature against
the user pool's published keys (JWKS), its issuer, its audience (the app
client), its expiry, and that it is an ID token, then takes the user id from
its `sub` claim. A caller can't name another user, so a misconfigured port no
longer means anyone can act as anyone.

Configured from the environment:

  COGNITO_ISSUER     https://cognito-idp.<region>.amazonaws.com/<user-pool-id>
  COGNITO_CLIENT_ID  the app client id (the token's audience)

AUTH_DISABLED=1 instead trusts an X-User-Id header. That is for local single-user
mode and tests only. With neither set, every request is refused: the API fails
closed rather than trusting whoever calls it.
"""

from __future__ import annotations

import os
import threading

import jwt

# Allow a little clock difference between Cognito and this server
LEEWAY_SECONDS = 30


class AuthError(Exception):
    """The request can't be tied to a user. The message is safe to show."""

    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.status = status


def header_trusted() -> bool:
    """Local mode: the caller's X-User-Id is taken at its word."""
    return os.getenv("AUTH_DISABLED") == "1"


def issuer() -> str | None:
    value = os.getenv("COGNITO_ISSUER", "").strip().rstrip("/")
    return value or None


def client_id() -> str | None:
    return os.getenv("COGNITO_CLIENT_ID", "").strip() or None


_clients: dict[str, jwt.PyJWKClient] = {}
_clients_lock = threading.Lock()


def _jwks_client(iss: str) -> jwt.PyJWKClient:
    """One cached key client per issuer; it refetches keys when Cognito rotates them."""
    with _clients_lock:
        if iss not in _clients:
            _clients[iss] = jwt.PyJWKClient(f"{iss}/.well-known/jwks.json", cache_keys=True)
        return _clients[iss]


def verify(authorization: str | None) -> dict:
    """The claims of a valid Cognito ID token in an Authorization header, or AuthError."""
    iss, aud = issuer(), client_id()
    if not iss or not aud:
        raise AuthError("Sign-in verification isn't configured on the server.", status=503)
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Missing sign-in token.")
    token = token.strip()
    try:
        key = _jwks_client(iss).get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token, key, algorithms=["RS256"], audience=aud, issuer=iss,
            leeway=LEEWAY_SECONDS, options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Your sign-in has expired. Please sign in again.") from exc
    except (jwt.PyJWTError, jwt.PyJWKClientError) as exc:
        raise AuthError("Invalid sign-in token.") from exc
    if claims.get("token_use") != "id":
        raise AuthError("Invalid sign-in token.")   # e.g. an access token
    return claims
