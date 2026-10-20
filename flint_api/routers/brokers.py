"""Broker connections: link a broker login, read its holdings and positions
live, disconnect. Read-only by design; nothing here can place an order.

Nothing about a broker is stored in the database or configured on the
server. Each user brings their own broker developer app (its keys) or a
read-only token, and each connection is a sealed cookie in the user's
browser (flint_api/broker_jar.py), kept for as long as the user chose.
Holdings and positions are fetched from the broker whenever they are shown.

The connect round trip:
  1. POST /brokers/{broker}/connect with the user's app keys and chosen
     expiry (or the id of a connection to reconnect, whose keys are reused)
     -> {redirect_url}. A sealed one-time state cookie (10 minutes) holds them.
  2. The broker redirects the browser to the frontend's callback with its
     one-time login values (and our state, where the broker echoes it).
  3. The frontend POSTs them to /brokers/{broker}/session through the BFF. The
     state cookie must be one this user created in this browser for this
     broker and not yet used, and must match the echoed state if there is one;
     that stops a crafted link from attaching someone else's broker login.
"""
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field, StringConstraints

import data_source

from ..broker_jar import MAX_CONNECTIONS, MAX_KEEP, SESSION_KEEP, Connection, Jar
from ..brokers import ADAPTERS, TokenExpired
from ..errors import AppError, Forbidden, NotFound, Unavailable
from ..ratelimit import limit
from ..security import Principal
from ..settings import settings

router = APIRouter(prefix="/brokers", tags=["brokers"])

SCOPES = ["profile:read", "holdings:read", "positions:read"]
# Missing last prices filled from a quote (Dhan sends none).
MAX_PRICE_LOOKUPS = 100

BrokerName = Literal["zerodha", "upstox", "groww", "angel", "fyers", "dhan", "indmoney", "binance"]
State = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{32,128}$")]
ParamName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_]{1,32}$")]
# Request tokens, auth codes, token ids and JWTs.
ParamValue = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._-]{1,4096}$")]
# Pasted access tokens and API keys (some are base64 or JWTs).
Credential = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9._\-+/=]{8,2048}$")]
# Broker-app keys, ids and secrets. Short, so a connection fits in one cookie.
AppValue = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9._\-+/=]{4,256}$")]
RedirectUri = Annotated[str, StringConstraints(max_length=300)]
# How long to keep a connection, like a GitHub token's expiration.
Keep = Literal["session", "token", "7d", "30d", "90d", "custom"]
_KEEP_DAYS = {"7d": 7, "30d": 30, "90d": 90}


class Keeping(BaseModel):
    keep: Keep = "token"
    # Only with keep="custom": the end date, at most MAX_KEEP away.
    keep_until: datetime | None = None


class ConnectBody(Keeping):
    # Polaris's callback page for this broker, as saved on the user's broker app.
    redirect_uri: RedirectUri
    # A new connection: the user's broker-app keys.
    app: Annotated[dict[ParamName, AppValue], Field(max_length=4)] | None = None
    # Or a reconnect: the connection whose app keys and expiry are reused.
    connection_id: uuid.UUID | None = None


class SessionBody(BaseModel):
    # Absent for brokers that don't echo it (Dhan); the state cookie still applies.
    state: State | None = None
    params: Annotated[dict[ParamName, ParamValue], Field(min_length=1, max_length=8)]


class LinkBody(Keeping):
    credentials: Annotated[dict[ParamName, Credential], Field(min_length=1, max_length=4)]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ready() -> bool:
    return bool(settings.broker_token_key)


def label(c: Connection) -> str:
    return ADAPTERS[c.broker].LABEL


def view(c: Connection, *, status: str | None = None, error: str | None = None) -> dict:
    """Never includes the access token or the app keys."""
    return {"id": c.id, "broker": c.broker, "label": label(c), "broker_user_id": c.broker_user_id,
            "broker_user_name": c.broker_user_name, "status": status or c.status, "account_id": c.id,
            "currency": ADAPTERS[c.broker].CURRENCY, "auth": ADAPTERS[c.broker].AUTH,
            "token_expires_at": c.expires_at, "connected_at": c.connected_at, "last_error": error,
            "kept_until": c.keep_until, "session_only": c.session_only, "scopes": SCOPES}


def callback_path(broker: str) -> str:
    # Zerodha's was registered before other brokers existed.
    return "/settings/brokers/callback" + ("" if broker == "zerodha" else f"/{broker}")


def _check_redirect(broker: str, uri: str) -> str:
    """Only Polaris's own callback page, on an origin the API already trusts:
    a login can't be sent anywhere else."""
    if not any(uri == origin.rstrip("/") + callback_path(broker) for origin in settings.authorized_parties):
        raise AppError("That isn't Polaris's callback address.")
    return uri


def _keep_rule(body: Keeping) -> dict:
    """The chosen expiry as stored with a pending login: a mode and, except for
    'token' (known once the broker answers), the end date."""
    now = _now()
    if body.keep == "custom":
        until = body.keep_until
        if until is None or until.tzinfo is None:
            raise AppError("Pick a date to keep the connection until.")
        if until <= now or until > now + MAX_KEEP:
            raise AppError(f"Pick a date within the next {MAX_KEEP.days} days.")
        return {"m": "until", "k": until.astimezone(timezone.utc).isoformat()}
    if body.keep == "session":
        return {"m": "session", "k": (now + SESSION_KEEP).isoformat()}
    if body.keep == "token":
        return {"m": "token"}
    return {"m": "until", "k": (now + timedelta(days=_KEEP_DAYS[body.keep])).isoformat()}


def _fill_prices(rows: list[dict]) -> None:
    """Best effort: a missing last price is taken from a quote, so values
    aren't shown at cost. Rows stay unpriced if the quote is unavailable."""
    cache: dict[str, Decimal | None] = {}
    for r in rows:
        if r["last_price"] is not None:
            continue
        if r["symbol"] not in cache and len(cache) < MAX_PRICE_LOOKUPS:
            try:
                price = (data_source.get_quote(r["symbol"]) or {}).get("regularMarketPrice")
            except Exception:
                price = None
            cache[r["symbol"]] = Decimal(str(price)) if isinstance(price, (int, float)) and price > 0 else None
        r["last_price"] = cache.get(r["symbol"])


def live(jar: Jar, c: Connection, *, positions: bool) -> tuple[list[dict], list[dict]]:
    """Holdings (and optionally today's positions) straight from the broker.
    An expired session drops the token from the cookie and is re-raised."""
    if c.status != "connected":
        raise TokenExpired(f"Your {label(c)} session has ended. Reconnect to see it again.", headers=jar.headers())
    adapter = ADAPTERS[c.broker]
    try:
        holdings, _ = adapter.holdings(c.access_token, c.app)
        open_positions = adapter.positions(c.access_token, c.app)[0] if positions else []
    except TokenExpired:
        c.access_token = None
        jar.save(c)
        raise TokenExpired(f"Your {label(c)} session has ended. Reconnect to see it again.", headers=jar.headers())
    _fill_prices(holdings + open_positions)
    return holdings, open_positions


@router.get("")
def list_connections(request: Request, response: Response, user: Principal = Depends(limit("standard"))):
    jar = Jar.of(request, user.user_id)
    connections = [view(c) for c in jar.connections()]
    response.headers.update(jar.headers())  # deletes expired or unreadable cookies
    return {"connections": connections,
            "max_keep_days": MAX_KEEP.days,
            "available": [{"broker": a.NAME, "label": a.LABEL, "markets": ["crypto" if a.CURRENCY == "USD" else "india"],
                           "configured": _ready(), "auth": a.AUTH, "fields": a.FIELDS,
                           "help": a.HELP, "help_url": a.HELP_URL,
                           "callback_path": callback_path(a.NAME) if a.AUTH == "oauth" else None}
                          for a in ADAPTERS.values()]}


@router.get("/watchlist")
def watchlist(request: Request, response: Response, user: Principal = Depends(limit("expensive"))):
    """One read-only list per connection: its holdings and today's positions,
    live from the broker. Separate from the user's own watchlists."""
    jar = Jar.of(request, user.user_id)
    lists = []
    for c in jar.connections():
        items: dict[str, dict] = {}
        status, error = None, None
        try:
            holdings, positions = live(jar, c, positions=True)
        except TokenExpired:
            holdings, positions, status = [], [], "needs_reconnect"
        except AppError as e:
            holdings, positions, status, error = [], [], "error", e.message
        for h in holdings:
            items[h["symbol"]] = {"symbol": h["symbol"], "broker_symbol": h["broker_symbol"], "exchange": h["exchange"],
                                  "held": format(h["quantity"].normalize(), "f"), "position": None}
        for p in positions:
            item = items.setdefault(p["symbol"], {"symbol": p["symbol"], "broker_symbol": p["broker_symbol"],
                                                  "exchange": p["exchange"], "held": None, "position": None})
            total = Decimal(item["position"] or 0) + p["quantity"]
            item["position"] = format(total.normalize(), "f")
        lists.append({"connection": view(c, status=status, error=error), "as_of": _now(),
                      "items": sorted(items.values(), key=lambda i: i["symbol"])})
    response.headers.update(jar.headers())
    return {"lists": lists}


@router.post("/{broker}/connect")
def connect(broker: BrokerName, body: ConnectBody, request: Request, response: Response,
            user: Principal = Depends(limit("write"))):
    adapter = _oauth(broker)
    if not _ready():
        raise Unavailable("Broker connections aren't available right now.")
    redirect_uri = _check_redirect(broker, body.redirect_uri)
    jar = Jar.of(request, user.user_id)
    if body.connection_id is not None:
        c = jar.get(str(body.connection_id))
        if c is None or c.broker != broker:
            raise NotFound("Connection not found. Connect it again with your app keys.", headers=jar.headers())
        pending = {"a": c.app, "m": "session" if c.session_only else "until", "k": c.keep_until.isoformat()}
    else:
        if set(body.app or {}) != {f["name"] for f in adapter.FIELDS}:
            raise AppError(f"{adapter.LABEL} needs: {', '.join(f['label'] for f in adapter.FIELDS)}.")
        pending = {"a": dict(body.app), **_keep_rule(body)}
    pending["r"] = redirect_uri
    try:
        redirect_url = adapter.login_url(jar.issue_state(broker, pending), pending["a"], redirect_uri)
    except AppError as e:
        e.headers.update(jar.headers())
        raise
    response.headers.update(jar.headers())
    return {"redirect_url": redirect_url}


@router.post("/{broker}/session", status_code=201)
def create_session(broker: BrokerName, body: SessionBody, request: Request, response: Response,
                   user: Principal = Depends(limit("write"))):
    adapter = _oauth(broker)
    jar = Jar.of(request, user.user_id)
    pending = jar.take_state(broker, body.state)
    if pending is None:
        raise Forbidden("This connection link has expired or was already used. Start again from Settings.",
                        headers=jar.headers())

    try:
        session = adapter.create_session(body.params, pending.get("a") or {}, pending.get("r") or "")
        c = _keep(jar, broker, session, pending.get("a") or {}, pending)
    except AppError as e:
        e.headers.update(jar.headers())  # the used state cookie is still cleared
        raise
    response.headers.update(jar.headers())
    return {"connection": view(c)}


@router.post("/{broker}/link", status_code=201)
def link(broker: BrokerName, body: LinkBody, request: Request, response: Response,
         user: Principal = Depends(limit("write"))):
    """Connects a broker without a login flow from a read-only token or API key
    the user created there. The adapter checks it with the broker first; it is
    kept only in this browser's sealed cookie."""
    adapter = ADAPTERS[broker]
    if adapter.AUTH != "keys":
        raise AppError(f"{adapter.LABEL} connects through its own login page.")
    if not _ready():
        raise Unavailable("Broker connections aren't available right now.")
    wanted = {f["name"] for f in adapter.FIELDS}
    if set(body.credentials) != wanted:
        raise AppError(f"{adapter.LABEL} needs: {', '.join(f['label'] for f in adapter.FIELDS)}.")
    rule = _keep_rule(body)
    jar = Jar.of(request, user.user_id)
    session = adapter.create_session(body.credentials)
    c = _keep(jar, broker, session, {}, rule)
    response.headers.update(jar.headers())
    return {"connection": view(c)}


def _oauth(broker: str):
    adapter = ADAPTERS[broker]
    if adapter.AUTH != "oauth":
        raise AppError(f"{adapter.LABEL} connects with a token or API key you paste in Settings.")
    return adapter


def _keep(jar: Jar, broker: str, session, app: dict, rule: dict) -> Connection:
    """Saves a new session as a connection cookie, reusing the id when the
    same broker login is connected again. `rule` is the chosen expiry
    (_keep_rule): 'token' keeps it until the broker session ends."""
    if rule.get("m") == "token":
        keep_until = min(session.expires_at, _now() + MAX_KEEP)
    else:
        keep_until = datetime.fromisoformat(rule["k"])
    existing = jar.connections()
    same = next((c for c in existing if c.broker == broker and c.broker_user_id == session.user_id), None)
    if same is None and len(existing) >= MAX_CONNECTIONS:
        raise AppError(f"You can connect up to {MAX_CONNECTIONS} broker logins.", headers=jar.headers())
    c = Connection(id=same.id if same else str(uuid.uuid4()), broker=broker, broker_user_id=session.user_id,
                   broker_user_name=session.user_name, access_token=session.access_token,
                   expires_at=session.expires_at, connected_at=same.connected_at if same else _now(),
                   keep_until=keep_until, session_only=rule.get("m") == "session", app=app)
    jar.save(c)
    return c


@router.delete("/{connection_id}", status_code=204)
def disconnect(connection_id: uuid.UUID, request: Request, response: Response,
               user: Principal = Depends(limit("write"))):
    """Forgets the connection, its token and app keys in this browser, and ends
    the broker session."""
    jar = Jar.of(request, user.user_id)
    c = jar.get(str(connection_id))
    if c is None:
        raise NotFound("Connection not found.", headers=jar.headers())
    jar.remove(c)
    if c.access_token:
        ADAPTERS[c.broker].logout(c.access_token, c.broker_user_id, c.app)
    response.headers.update(jar.headers())
