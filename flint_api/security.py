"""Clerk session verification.

Every route except /api/health depends on `current_user`. The token is a
Clerk session JWT (RS256), sent as `Authorization: Bearer <token>` by the
Next.js server, never by browser code. We verify signature, expiry, issuer
and the authorised party ourselves; nothing about the caller is trusted
until that passes.
"""
from dataclasses import dataclass

import jwt
from fastapi import Request

from .errors import NotFound, Unauthorized
from .settings import settings

_ALGORITHMS = ["RS256"]
_LEEWAY_SECONDS = 5
_jwk_client: jwt.PyJWKClient | None = None


@dataclass(frozen=True)
class Principal:
    user_id: str
    session_id: str | None = None


def _signing_key(token: str):
    if settings.clerk_jwt_key:
        return settings.clerk_jwt_key
    if not settings.clerk_jwks_url:
        # Fail closed: an unconfigured server must not accept anyone.
        raise Unauthorized("Authentication is not configured.")
    global _jwk_client
    if _jwk_client is None:
        _jwk_client = jwt.PyJWKClient(settings.clerk_jwks_url, cache_keys=True,
                                      lifespan=3600, timeout=5)
    return _jwk_client.get_signing_key_from_jwt(token).key


def verify_token(token: str) -> Principal:
    try:
        claims = jwt.decode(
            token,
            _signing_key(token),
            algorithms=_ALGORITHMS,
            issuer=settings.clerk_issuer or None,
            leeway=_LEEWAY_SECONDS,
            options={"require": ["exp", "iat", "sub", "iss"]},
        )
    except jwt.PyJWTError:
        raise Unauthorized("Your session is invalid or has expired. Sign in again.")

    azp = claims.get("azp")
    if azp and azp not in settings.authorized_parties:
        raise Unauthorized("Your session is invalid or has expired. Sign in again.")
    # Clerk marks a session with outstanding tasks (e.g. required MFA setup)
    # as "pending"; only fully active sessions may use the API.
    if claims.get("sts") not in (None, "active"):
        raise Unauthorized("Finish signing in to continue.")

    return Principal(user_id=claims["sub"], session_id=claims.get("sid"))


def require_admin(principal: Principal) -> Principal:
    if principal.user_id not in settings.admin_user_ids:
        # Same answer as a missing route: don't confirm operator surfaces exist.
        raise NotFound("Not found.")
    return principal


def current_user(request: Request) -> Principal:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Unauthorized("Sign in to continue.")
    principal = verify_token(token.strip())
    request.state.user_id = principal.user_id
    return principal
