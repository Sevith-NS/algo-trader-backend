"""Broker connections live in the user's browser, never in the database.

Each connection (broker, client id, access token, the user's own broker app
keys, how long to keep them) is sealed with AES-GCM (secretbox) into its own
httpOnly cookie, `pb_<connection id>`, bound to the Clerk user and the cookie
name: a cookie copied to another user or renamed does not open. The pending
login state, with the app keys the user just typed, is a short-lived
`pb_state` cookie sealed the same way.

The user picks how long a connection is kept, like a GitHub token's
expiration: this browser session, until the broker session ends, a number of
days, or a date, never more than MAX_KEEP. The deadline is inside the sealed
payload and checked on every request; the cookie's Max-Age only mirrors it.

The browser never talks to this API directly. The frontend's BFF forwards the
`pb_*` cookies in the X-Broker-Sessions request header and turns the
X-Broker-Cookies response header (JSON) back into Set-Cookie, so no script
in the page can read a token or key.
"""
import base64
import binascii
import hmac
import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from fastapi import Request

from . import secretbox
from .errors import AppError

REQUEST_HEADER = "x-broker-sessions"
RESPONSE_HEADER = "x-broker-cookies"
PREFIX = "pb_"
STATE_COOKIE = "pb_state"
STATE_TTL = timedelta(minutes=10)
# The longest a user can ask Polaris to keep a connection and its app keys.
MAX_KEEP = timedelta(days=90)
# "This browser session" still ends here, for browsers that restore sessions.
SESSION_KEEP = timedelta(hours=24)
MAX_CONNECTIONS = 5
# The BFF drops longer values (browsers keep ~4096 bytes per cookie).
MAX_COOKIE_VALUE = 3900

_NAME = re.compile(r"^pb_(state|[0-9a-f]{32})$")
_MAX_COOKIES = MAX_CONNECTIONS + 4


@dataclass
class Connection:
    id: str
    broker: str
    broker_user_id: str
    broker_user_name: str | None
    access_token: str | None
    expires_at: datetime
    connected_at: datetime
    keep_until: datetime
    # True: a browser-session cookie, gone when the browser closes.
    session_only: bool = False
    # The user's own developer-app keys at the broker; empty for pasted tokens.
    app: dict = field(default_factory=dict)

    @property
    def cookie(self) -> str:
        return PREFIX + self.id.replace("-", "")

    @property
    def status(self) -> str:
        if self.access_token is None or self.expires_at <= datetime.now(timezone.utc):
            return "needs_reconnect"
        return "connected"


def _b64(blob: bytes) -> str:
    return base64.urlsafe_b64encode(blob).decode().rstrip("=")


def _unb64(text: str) -> bytes | None:
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        return None


class Jar:
    """The caller's broker cookies for one request, plus the changes to send back."""

    def __init__(self, user_id: str, raw: str):
        self.user_id = user_id
        self._cookies: dict[str, str] = {}
        for part in raw.split(";")[:_MAX_COOKIES]:
            name, _, value = part.strip().partition("=")
            if _NAME.match(name) and value:
                self._cookies[name] = value
        self._out: list[dict] = []

    @classmethod
    def of(cls, request: Request, user_id: str) -> "Jar":
        return cls(user_id, request.headers.get(REQUEST_HEADER, ""))

    def _open(self, name: str) -> dict | None:
        blob = _unb64(self._cookies.get(name, ""))
        text = blob and secretbox.open_(blob, user_id=self.user_id, row_id=name)
        try:
            return json.loads(text) if text else None
        except ValueError:
            return None

    def _set(self, name: str, payload: dict | None, max_age: int | None = 0) -> None:
        """max_age None makes a browser-session cookie."""
        value = None if payload is None else _b64(secretbox.seal(json.dumps(payload), user_id=self.user_id, row_id=name))
        if value is not None and len(value) > MAX_COOKIE_VALUE:
            raise AppError("That login is too large to keep in this browser. Check the keys you entered.",
                           headers=self.headers())
        self._out = [c for c in self._out if c["name"] != name]
        self._out.append({"name": name, "value": value, "max_age": max_age if value else 0})
        if value is None:
            self._cookies.pop(name, None)
        else:
            self._cookies[name] = value

    # ------------------------------------------------------------ connections --

    def connections(self) -> list[Connection]:
        """Open connections. One past its keep-until date is deleted, as is any
        cookie that no longer opens (wrong user, old format, tampered)."""
        now = datetime.now(timezone.utc)
        out = []
        for name in list(self._cookies):
            if name == STATE_COOKIE:
                continue
            p = self._open(name)
            try:
                c = Connection(id=p["id"], broker=p["b"], broker_user_id=p["u"], broker_user_name=p.get("n"),
                               access_token=p.get("t"), expires_at=datetime.fromisoformat(p["e"]),
                               connected_at=datetime.fromisoformat(p["c"]),
                               keep_until=datetime.fromisoformat(p["k"]), session_only=bool(p.get("o")),
                               app=dict(p.get("a") or {}))
            except (KeyError, TypeError, ValueError):
                self._set(name, None)
                continue
            if c.keep_until <= now:
                self._set(name, None)
                continue
            out.append(c)
        return sorted(out, key=lambda c: c.connected_at)

    def get(self, connection_id: str) -> Connection | None:
        return next((c for c in self.connections() if c.id == connection_id), None)

    def save(self, c: Connection) -> None:
        left = int((c.keep_until - datetime.now(timezone.utc)).total_seconds())
        if left <= 0:
            self.remove(c)
            return
        self._set(c.cookie, {"id": c.id, "b": c.broker, "u": c.broker_user_id, "n": c.broker_user_name,
                             "t": c.access_token, "e": c.expires_at.isoformat(), "c": c.connected_at.isoformat(),
                             "k": c.keep_until.isoformat(), "o": c.session_only, "a": c.app},
                  None if c.session_only else left)

    def remove(self, c: Connection) -> None:
        self._set(c.cookie, None)

    # ------------------------------------------------------------- link state --

    def issue_state(self, broker: str, pending: dict) -> str:
        """`pending` is what the login needs once the broker sends the user back
        (the app keys, the chosen expiry, the redirect URL); sealed with the state."""
        state = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + STATE_TTL
        self._set(STATE_COOKIE, {"s": state, "b": broker, "e": expires.isoformat(), "p": pending},
                  int(STATE_TTL.total_seconds()))
        return state

    def take_state(self, broker: str, echoed: str | None) -> dict | None:
        """The pending login, once per connect started in this browser by this
        user, for this broker, within the TTL; otherwise None. If the broker
        echoed a state, it must match."""
        p = self._open(STATE_COOKIE)
        if STATE_COOKIE in self._cookies:
            self._set(STATE_COOKIE, None)
        if not p or p.get("b") != broker or not isinstance(p.get("p"), dict):
            return None
        try:
            if datetime.fromisoformat(p["e"]) <= datetime.now(timezone.utc):
                return None
        except (KeyError, TypeError, ValueError):
            return None
        if echoed is not None and not hmac.compare_digest(str(p.get("s", "")), echoed):
            return None
        return p["p"]

    # --------------------------------------------------------------- response --

    def headers(self) -> dict:
        return {RESPONSE_HEADER: json.dumps(self._out)} if self._out else {}
