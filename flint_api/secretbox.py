"""Encryption at rest for broker access tokens.

AES-256-GCM with a random 96-bit nonce per seal. The associated data binds a
ciphertext to its owner and row, so a token copied into someone else's row
fails to open instead of granting access to the first user's broker account.
"""
import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import Unavailable
from .settings import settings

_VERSION = b"\x01"


def _cipher() -> AESGCM:
    try:
        key = base64.b64decode(settings.broker_token_key, validate=True)
    except ValueError:
        key = b""
    if len(key) != 32:
        raise Unavailable("Broker connections aren't available right now.")
    return AESGCM(key)


def _aad(user_id: str, row_id: str) -> bytes:
    return f"{user_id}|{row_id}".encode()


def seal(plaintext: str, *, user_id: str, row_id: str) -> bytes:
    nonce = os.urandom(12)
    return _VERSION + nonce + _cipher().encrypt(nonce, plaintext.encode(), _aad(user_id, row_id))


def open_(blob: bytes, *, user_id: str, row_id: str) -> str | None:
    """The plaintext, or None if the blob is unreadable (wrong key, row or user)."""
    blob = bytes(blob)
    if len(blob) < 1 + 12 + 16 or blob[:1] != _VERSION:
        return None
    try:
        return _cipher().decrypt(blob[1:13], blob[13:], _aad(user_id, row_id)).decode()
    except InvalidTag:
        return None
