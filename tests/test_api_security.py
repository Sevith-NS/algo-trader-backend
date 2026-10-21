"""Edge security: authentication, error contract, headers, body cap, limits."""
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from conftest import auth, make_token


def test_health_is_public(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_missing_token_is_401(client):
    r = client.get("/api/session")
    assert r.status_code == 401
    assert r.json()["code"] == "unauthorized"


def test_valid_token_identifies_user(client):
    r = client.get("/api/session", headers=auth("user_abc"))
    assert r.status_code == 200 and r.json() == {"user_id": "user_abc"}


def test_token_signed_by_another_key_is_rejected(client):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    r = client.get("/api/session", headers={"Authorization": f"Bearer {make_token(key=other)}"})
    assert r.status_code == 401


def test_expired_token_is_rejected(client):
    r = client.get("/api/session", headers=auth(exp_in=-60))
    assert r.status_code == 401


def test_wrong_issuer_is_rejected(client):
    r = client.get("/api/session", headers=auth(iss="https://evil.example"))
    assert r.status_code == 401


def test_unknown_authorized_party_is_rejected(client):
    r = client.get("/api/session", headers=auth(azp="https://evil.example"))
    assert r.status_code == 401


def test_pending_session_is_rejected(client):
    r = client.get("/api/session", headers=auth(sts="pending"))
    assert r.status_code == 401


def test_alg_none_is_rejected(client):
    import base64
    import json
    seg = lambda o: base64.urlsafe_b64encode(json.dumps(o).encode()).rstrip(b"=").decode()
    token = f'{seg({"alg": "none", "typ": "JWT"})}.{seg({"sub": "x", "iss": "y", "exp": 9999999999, "iat": 1})}.'
    r = client.get("/api/session", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_security_headers_and_no_cors(client):
    r = client.get("/api/health", headers={"Origin": "https://evil.example"})
    h = r.headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert "default-src 'none'" in h["content-security-policy"]
    assert h["cache-control"] == "private, no-store"
    assert "x-request-id" in h
    assert "access-control-allow-origin" not in h


def test_unknown_route_is_json_404(client):
    r = client.get("/api/nope", headers=auth())
    assert r.status_code == 404 and r.json()["code"] == "not_found"


def test_oversized_body_is_413(client):
    r = client.post("/api/session", content=b"x" * (1024 * 1024 + 1), headers=auth())
    assert r.status_code == 413


def test_feed_links_keep_only_http():
    import data_source
    assert data_source.web_link("https://example.com/a") == "https://example.com/a"
    assert data_source.web_link(" http://example.com ") == "http://example.com"
    for bad in ("javascript:alert(1)", "JaVaScRiPt:x", "data:text/html,x", "//evil.com", "", None, 42):
        assert data_source.web_link(bad) is None


def test_previous_close_matches_by_session_date():
    import data_source
    # NYSE, gmtoffset -4h. Candles open 13:30 UTC; the quote time is that day's close.
    rec = [{"t": "2026-10-08T13:30:00+00:00", "c": 340.42},
           {"t": "2026-10-09T13:30:00+00:00", "c": 336.6400146484375}]
    meta = {"regularMarketTime": 1791576000, "gmtoffset": -14400}   # 2026-10-09 16:00 EDT
    # Float32 candle close vs a rounded price must still count as the same session.
    assert data_source._previous_close(rec, meta, 336.64) == 340.42
    # Quote from a later session whose candle hasn't arrived: the last candle is the previous close.
    assert data_source._previous_close(rec, {**meta, "regularMarketTime": 1791576000 + 3 * 86400}, 338.0) == 336.6400146484375
