"""Broker connections: link a Zerodha login, sync its holdings, disconnect.

Read-only by design. A connection creates one `real` account whose holdings
are replaced on each sync; nothing here can place an order.

The connect round trip:
  1. POST /brokers/zerodha/connect   -> {redirect_url}; a one-time state nonce
     is stored (hashed) for 10 minutes.
  2. Kite redirects the browser to the frontend's /settings/brokers/callback
     with ?request_token=...&state=...
  3. The frontend POSTs both to /brokers/zerodha/session through the BFF, so
     the exchange runs with the user's verified session. The state must be one
     this user created and not yet used; that stops a crafted link from
     attaching someone else's broker login to the victim's account.
"""
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, StringConstraints

from .. import db, kite, secretbox
from ..errors import AppError, Forbidden, NotFound, Unavailable
from ..ratelimit import limit
from ..security import Principal
from ..settings import settings

router = APIRouter(prefix="/brokers", tags=["brokers"])

MAX_CONNECTIONS = 5
STATE_TTL = timedelta(minutes=10)
SCOPES = ["profile:read", "holdings:read"]
_IST = timezone(timedelta(hours=5, minutes=30))

RequestToken = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9]{8,64}$")]
State = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{32,128}$")]


class SessionBody(BaseModel):
    request_token: RequestToken
    state: State


def _hash(state: str) -> bytes:
    return hashlib.sha256(state.encode()).digest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _view(row: dict) -> dict:
    status = row["status"]
    expires = row["token_expires_at"]
    if status == "connected" and (row["access_token_enc"] is None or (expires and expires <= _now())):
        status = "needs_reconnect"
    return {"id": str(row["id"]), "broker": row["broker"], "broker_user_id": row["broker_user_id"],
            "broker_user_name": row["broker_user_name"], "status": status,
            "account_id": str(row["account_id"]) if row.get("account_id") else None,
            "last_synced_at": row["last_synced_at"], "token_expires_at": expires,
            "last_error": row["last_error"], "scopes": SCOPES, "created_at": row["created_at"]}


_SELECT = ("select c.*, a.id as account_id from broker_connections c "
           "left join accounts a on a.broker_connection_id = c.id")


@router.get("")
def list_connections(user: Principal = Depends(limit("standard"))):
    with db.user_tx(user.user_id) as conn:
        rows = conn.execute(_SELECT + " order by c.created_at").fetchall()
    return {"connections": [_view(r) for r in rows],
            "available": [{"broker": "zerodha", "label": "Zerodha", "markets": ["india"],
                           "configured": settings.zerodha_configured}]}


@router.post("/zerodha/connect")
def connect(user: Principal = Depends(limit("write"))):
    if not settings.zerodha_configured:
        raise Unavailable("Zerodha isn't set up on this server yet.")
    state = secrets.token_urlsafe(32)
    with db.user_tx(user.user_id) as conn:
        conn.execute("delete from broker_link_states where expires_at <= now()")
        conn.execute("insert into broker_link_states (state_hash, user_id, broker, expires_at) "
                     "values (%s, %s, 'zerodha', %s)", (_hash(state), user.user_id, _now() + STATE_TTL))
    return {"redirect_url": kite.login_url(state), "expires_in": int(STATE_TTL.total_seconds())}


@router.post("/zerodha/session", status_code=201)
def create_session(body: SessionBody, user: Principal = Depends(limit("write"))):
    with db.user_tx(user.user_id) as conn:
        used = conn.execute("delete from broker_link_states where state_hash = %s and broker = 'zerodha' "
                            "and expires_at > now() returning 1", (_hash(body.state),)).fetchone()
    if used is None:
        raise Forbidden("This connection link has expired or was already used. Start again from Settings.")

    session = kite.create_session(body.request_token)  # network, outside any transaction

    with db.user_tx(user.user_id) as conn:
        row = conn.execute("select id from broker_connections where broker = 'zerodha' and broker_user_id = %s",
                           (session["user_id"],)).fetchone()
        if row is None:
            if conn.execute("select count(*) as n from broker_connections").fetchone()["n"] >= MAX_CONNECTIONS:
                raise AppError(f"You can connect up to {MAX_CONNECTIONS} broker logins.")
            conn_id = str(uuid.uuid4())
            conn.execute("insert into broker_connections (id, user_id, broker, broker_user_id, broker_user_name, "
                         "access_token_enc, token_expires_at) values (%s, %s, 'zerodha', %s, %s, %s, %s)",
                         (conn_id, user.user_id, session["user_id"], session["user_name"],
                          secretbox.seal(session["access_token"], user_id=user.user_id, row_id=conn_id),
                          kite.session_expiry()))
            conn.execute("insert into accounts (user_id, name, currency, kind, broker_connection_id) "
                         "values (%s, %s, 'INR', 'real', %s)",
                         (user.user_id, f"Zerodha {session['user_id']}"[:40], conn_id))
        else:
            conn_id = str(row["id"])
            conn.execute("update broker_connections set status = 'connected', broker_user_name = %s, "
                         "access_token_enc = %s, token_expires_at = %s, last_error = null where id = %s",
                         (session["user_name"],
                          secretbox.seal(session["access_token"], user_id=user.user_id, row_id=conn_id),
                          kite.session_expiry(), conn_id))

    try:
        sync = _sync(user.user_id, conn_id)
    except AppError as e:
        sync = {"error": e.message}
    with db.user_tx(user.user_id) as conn:
        view = _view(conn.execute(_SELECT + " where c.id = %s", (conn_id,)).fetchone())
    return {"connection": view, "sync": sync}


@router.post("/{connection_id}/sync")
def sync(connection_id: UUID, user: Principal = Depends(limit("expensive"))):
    return _sync(user.user_id, str(connection_id))


@router.delete("/{connection_id}", status_code=204)
def disconnect(connection_id: UUID, user: Principal = Depends(limit("write"))):
    """Removes the connection, its real account, holdings and history."""
    with db.user_tx(user.user_id) as conn:
        row = conn.execute("delete from broker_connections where id = %s returning access_token_enc",
                           (connection_id,)).fetchone()
    if row is None:
        raise NotFound("Connection not found.")
    token = row["access_token_enc"] and secretbox.open_(row["access_token_enc"], user_id=user.user_id,
                                                         row_id=str(connection_id))
    if token:
        kite.logout(token)


def _mark(user_id: str, conn_id: str, status: str, error: str | None, *, drop_token: bool = False) -> None:
    with db.user_tx(user_id) as conn:
        conn.execute("update broker_connections set status = %s, last_error = %s"
                     + (", access_token_enc = null" if drop_token else "") + " where id = %s",
                     (status, error, conn_id))


def _sync(user_id: str, conn_id: str) -> dict:
    with db.user_tx(user_id) as conn:
        row = conn.execute(_SELECT + " where c.id = %s", (conn_id,)).fetchone()
    if row is None or row["account_id"] is None:
        raise NotFound("Connection not found.")
    expired = row["token_expires_at"] is not None and row["token_expires_at"] <= _now()
    token = None if expired or row["access_token_enc"] is None else secretbox.open_(
        row["access_token_enc"], user_id=user_id, row_id=conn_id)
    if token is None:
        _mark(user_id, conn_id, "needs_reconnect", None, drop_token=True)
        raise kite.TokenExpired("Your Zerodha session has ended. Reconnect to sync again.")

    try:
        rows, skipped = kite.holdings(token)
    except kite.TokenExpired:
        _mark(user_id, conn_id, "needs_reconnect", None, drop_token=True)
        raise
    except AppError as e:
        _mark(user_id, conn_id, "error", e.message[:300])
        raise

    account_id = row["account_id"]
    synced_at = _now()
    value = sum((r["quantity"] * (r["last_price"] or r["close_price"] or r["avg_price"]) for r in rows), Decimal(0))
    invested = sum((r["quantity"] * r["avg_price"] for r in rows), Decimal(0))
    with db.user_tx(user_id) as conn:
        conn.execute("delete from holdings where account_id = %s", (account_id,))
        for r in rows:
            conn.execute("insert into holdings (account_id, user_id, symbol, broker_symbol, exchange, isin, quantity, "
                         "avg_price, last_price, close_price, currency, synced_at) "
                         "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                         (account_id, user_id, r["symbol"], r["broker_symbol"], r["exchange"], r["isin"],
                          r["quantity"], r["avg_price"], r["last_price"], r["close_price"], r["currency"], synced_at))
        conn.execute("insert into account_snapshots (account_id, user_id, on_date, market_value, invested, currency) "
                     "values (%s, %s, %s, %s, %s, 'INR') on conflict (account_id, on_date) do update "
                     "set market_value = excluded.market_value, invested = excluded.invested",
                     (account_id, user_id, synced_at.astimezone(_IST).date(), value, invested))
        conn.execute("update broker_connections set status = 'connected', last_synced_at = %s, last_error = null "
                     "where id = %s", (synced_at, conn_id))
    return {"account_id": str(account_id), "holdings": len(rows), "skipped": skipped, "synced_at": synced_at}
