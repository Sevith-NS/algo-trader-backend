"""Test harness: a throwaway RSA key stands in for Clerk.

Environment is set before `flint_api` is imported, so the settings object
picks up the test key and issuer instead of anything in .env.clerk.
"""
import os
import sys
import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ISSUER = "https://test-instance.clerk.accounts.dev"
ORIGIN = "http://localhost:3000"

_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PRIVATE_PEM = _key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption())
PUBLIC_PEM = _key.public_key().public_bytes(serialization.Encoding.PEM,
                                            serialization.PublicFormat.SubjectPublicKeyInfo).decode()

os.environ["CLERK_ISSUER"] = ISSUER
os.environ["CLERK_JWT_KEY"] = PUBLIC_PEM
os.environ["AUTHORIZED_PARTIES"] = ORIGIN


def make_token(sub: str = "user_test_a", *, key=PRIVATE_PEM, iss: str = ISSUER,
               azp: str | None = ORIGIN, exp_in: int = 60, **extra) -> str:
    now = int(time.time())
    claims = {"sub": sub, "iss": iss, "iat": now, "nbf": now, "exp": now + exp_in,
              "sid": f"sess_{uuid.uuid4().hex[:8]}", **extra}
    if azp is not None:
        claims["azp"] = azp
    return jwt.encode(claims, key, algorithm="RS256")


def auth(sub: str = "user_test_a", **kw) -> dict:
    return {"Authorization": f"Bearer {make_token(sub, **kw)}"}


@pytest.fixture(scope="session", autouse=True)
def _close_pool():
    yield
    from flint_api import db
    db.close()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from flint_api import ratelimit
    from flint_api.main import app
    ratelimit.reset_for_tests()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
