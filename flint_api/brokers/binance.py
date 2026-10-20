"""Binance spot, read-only: balances.

Binance has no login flow for other apps: the user creates an API key in
Binance with only "Enable Reading" and pastes the key and secret here.
Before a key is kept, Binance's own apiRestrictions endpoint must show it can
read and can do nothing else: a key that can trade, transfer or withdraw is
refused. Requests are signed with HMAC-SHA256 as Binance specifies.
"""
import hashlib
import hmac
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import requests

from ..errors import AppError, Unavailable
from .base import Session, TokenExpired, dec, holding, param

NAME, LABEL = "binance", "Binance"
CURRENCY, AUTH = "USD", "keys"
FIELDS = [{"name": "api_key", "label": "API key", "secret": False},
          {"name": "api_secret", "label": "Secret key", "secret": True}]
HELP_URL = "https://www.binance.com/en/my/settings/api-management"
HELP = ("In Binance, open API Management and create a key with only “Enable Reading” ticked. "
        "Polaris refuses keys that can trade, transfer or withdraw.")
API_URL = "https://api.binance.com"
_TIMEOUT = 15
_KEY = re.compile(r"^[A-Za-z0-9]{16,128}$")
_ASSET = re.compile(r"^[A-Z0-9]{1,15}$")
# Any of these on a key means it can do more than read.
_BEYOND_READ = ("enableWithdrawals", "enableInternalTransfer", "enableMargin", "enableFutures",
                "permitsUniversalTransfer", "enableVanillaOptions", "enableSpotAndMarginTrading",
                "enablePortfolioMarginTrading", "enableFixApiTrade")
# Binance: invalid key format, invalid key/IP/permissions, bad signature.
_KEY_ERRORS = {-2014, -2015, -1022}



def _split(token: str) -> tuple[str, str]:
    key, _, secret = token.partition(":")
    return key, secret


def _signed(path: str, key: str, secret: str, params: dict | None = None):
    query = urlencode({**(params or {}), "timestamp": int(time.time() * 1000), "recvWindow": 5000})
    signature = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    try:
        resp = requests.request("GET", f"{API_URL}{path}?{query}&signature={signature}",
                                headers={"X-MBX-APIKEY": key}, timeout=_TIMEOUT)
    except requests.RequestException:
        raise Unavailable("Binance didn't respond. Try again in a moment.")
    try:
        body = resp.json()
    except ValueError:
        body = {}
    code = body.get("code") if isinstance(body, dict) else None
    if resp.status_code == 401 or code in _KEY_ERRORS:
        raise TokenExpired("Binance didn't accept that API key. Check the key, secret and IP restriction.")
    if resp.status_code >= 400:
        raise Unavailable("Binance couldn't complete the request. Try again in a moment.")
    return body


def create_session(params: dict) -> Session:
    key, secret = param(params, "api_key", LABEL), param(params, "api_secret", LABEL)
    if not _KEY.match(key) or not _KEY.match(secret):
        raise AppError("That doesn't look like a Binance API key and secret.")
    rights = _signed("/sapi/v1/account/apiRestrictions", key, secret)
    if not rights.get("enableReading"):
        raise AppError("That key can't read your account. Tick “Enable Reading” for it in Binance.")
    if any(rights.get(k) for k in _BEYOND_READ):
        raise AppError("That key can trade, transfer or withdraw. Create a key with only “Enable Reading” "
                       "and connect that one.")
    account = _signed("/api/v3/account", key, secret, {"omitZeroBalances": "true"})
    uid = account.get("uid")
    return Session(access_token=f"{key}:{secret}", user_id=str(uid) if uid else f"key …{key[-4:]}",
                   user_name=None, expires_at=datetime.now(timezone.utc) + timedelta(days=30))


def logout(access_token: str, broker_user_id: str, app: dict) -> None:
    """The key belongs to the user's Binance account; they can delete it there."""


def holdings(access_token: str, app: dict) -> tuple[list[dict], int]:
    """Spot balances, quoted as <ASSET>-USD. Binance keeps no purchase price
    for a balance, so cost and P&L are unknown (None)."""
    key, secret = _split(access_token)
    rows, skipped = [], 0
    account = _signed("/api/v3/account", key, secret, {"omitZeroBalances": "true"})
    for b in account.get("balances") or []:
        asset = b.get("asset") or ""
        qty = (dec(b.get("free")) or 0) + (dec(b.get("locked")) or 0)
        if not _ASSET.match(asset) or qty <= 0:
            skipped += 1
            continue
        rows.append(holding(f"{asset}-USD", asset, "BINANCE", None, qty, None, None, None, currency="USD"))
    return rows, skipped


def positions(access_token: str, app: dict) -> tuple[list[dict], int]:
    """Spot balances only; Binance has no day positions in this sense."""
    return [], 0
