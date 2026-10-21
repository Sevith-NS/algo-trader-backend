"""Price alerts: price crosses above/below a level, or the day's move passes a
percentage either way.

There is no background worker on this deployment, so alerts are evaluated
when the user's app asks (listing alerts or polling firings), at most once a
minute per user, against delayed quotes. An alert fires once, then sits in
`triggered` until the user re-arms it. Stale quotes never fire an alert.
"""
import threading
import time
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, StringConstraints

import data_source

from .. import db
from ..errors import AppError, NotFound
from ..ratelimit import limit
from ..security import Principal
from ..validators import Symbol

router = APIRouter(prefix="/alerts", tags=["alerts"])

MAX_ALERTS = 50
CHECK_EVERY = 60  # seconds, per user
CHECK_NOTE = "Checked about once a minute while Polaris is open, on delayed data."

Kind = Literal["price_above", "price_below", "move_pct"]
Threshold = Annotated[Decimal, Field(gt=0, le=Decimal("1e12"), max_digits=24, decimal_places=8)]
Note = Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)]

_lock = threading.Lock()
_last_check: dict[str, float] = {}


class CreateBody(BaseModel):
    symbol: Symbol
    kind: Kind
    threshold: Threshold
    note: Note | None = None


class UpdateBody(BaseModel):
    threshold: Threshold | None = None
    note: Note | None = None
    status: Literal["active", "paused"] | None = None


def _plain(d: Decimal) -> str:
    """No exponent: Decimal("100").normalize() would print as "1E+2"."""
    return format(d.normalize(), "f")


def _validate(kind: str, threshold: Decimal) -> None:
    if kind == "move_pct" and threshold > 100:
        raise AppError("A percentage move alert needs a value up to 100.")


def _row(r: dict) -> dict:
    return {"id": str(r["id"]), "symbol": r["symbol"], "kind": r["kind"], "threshold": _plain(r["threshold"]),
            "status": r["status"], "note": r["note"],
            "reference_price": _plain(r["reference_price"]) if r["reference_price"] is not None else None,
            "created_at": r["created_at"], "triggered_at": r["triggered_at"], "last_checked_at": r["last_checked_at"]}


def _fires(kind: str, threshold: float, price: float, change_pct: float | None) -> bool:
    if kind == "price_above":
        return price >= threshold
    if kind == "price_below":
        return price <= threshold
    return change_pct is not None and abs(change_pct) >= threshold


def evaluate(user_id: str, *, force: bool = False) -> None:
    """Check the user's active alerts against current quotes, at most once a minute."""
    now = time.monotonic()
    with _lock:
        if not force and now - _last_check.get(user_id, -CHECK_EVERY) < CHECK_EVERY:
            return
        _last_check[user_id] = now
        if len(_last_check) > 20_000:
            _last_check.clear()

    with db.user_tx(user_id) as conn:
        active = conn.execute("select id, symbol, kind, threshold from alerts where status = 'active'").fetchall()
    if not active:
        return

    quotes = {}
    for symbol in {a["symbol"] for a in active}:  # network, outside any transaction
        try:
            quotes[symbol] = data_source.get_quote(symbol) or {}
        except Exception:
            quotes[symbol] = {}

    with db.user_tx(user_id) as conn:
        for a in active:
            q = quotes[a["symbol"]]
            price = q.get("regularMarketPrice")
            if q.get("error") or q.get("stale") or not isinstance(price, (int, float)) or price <= 0:
                continue
            change = q.get("regularMarketChangePercent")
            change = float(change) if isinstance(change, (int, float)) else None
            if _fires(a["kind"], float(a["threshold"]), float(price), change):
                updated = conn.execute("update alerts set status = 'triggered', triggered_at = now(), "
                                       "last_checked_at = now() where id = %s and status = 'active' returning id",
                                       (a["id"],)).fetchone()
                if updated:
                    conn.execute("insert into alert_firings (alert_id, user_id, price, change_pct) "
                                 "values (%s, %s, %s, %s)",
                                 (a["id"], user_id, Decimal(str(price)),
                                  Decimal(str(round(change, 4))) if change is not None else None))
            else:
                conn.execute("update alerts set last_checked_at = now() where id = %s", (a["id"],))


@router.get("")
def list_alerts(user: Principal = Depends(limit("standard")),
                status: Literal["active", "paused", "triggered"] | None = None):
    evaluate(user.user_id)
    with db.user_tx(user.user_id) as conn:
        sql = "select * from alerts" + (" where status = %s" if status else "") + " order by created_at desc"
        rows = conn.execute(sql, (status,) if status else ()).fetchall()
    return {"alerts": [_row(r) for r in rows], "note": CHECK_NOTE}


@router.post("", status_code=201)
def create_alert(body: CreateBody, user: Principal = Depends(limit("write"))):
    _validate(body.kind, body.threshold)
    try:
        q = data_source.get_quote(body.symbol) or {}
    except Exception:
        q = {}
    ref = q.get("regularMarketPrice")
    ref = Decimal(str(ref)) if isinstance(ref, (int, float)) and ref > 0 and not q.get("stale") else None
    with db.user_tx(user.user_id) as conn:
        if conn.execute("select count(*) as n from alerts where status <> 'triggered'").fetchone()["n"] >= MAX_ALERTS:
            raise AppError(f"You can have up to {MAX_ALERTS} active or paused alerts.")
        row = conn.execute("insert into alerts (user_id, symbol, kind, threshold, note, reference_price) "
                           "values (%s, %s, %s, %s, %s, %s) returning *",
                           (user.user_id, body.symbol, body.kind, body.threshold, body.note or None, ref)).fetchone()
    return {**_row(row), "note_on_checks": CHECK_NOTE}


@router.patch("/{alert_id}")
def update_alert(alert_id: UUID, body: UpdateBody, user: Principal = Depends(limit("write"))):
    """Edit the level or note, pause, or re-arm (status=active) a triggered alert."""
    with db.user_tx(user.user_id) as conn:
        current = conn.execute("select * from alerts where id = %s", (alert_id,)).fetchone()
        if current is None:
            raise NotFound("Alert not found.")
        threshold = body.threshold if body.threshold is not None else current["threshold"]
        _validate(current["kind"], threshold)
        status = body.status or current["status"]
        note = current["note"] if body.note is None else (body.note or None)
        rearm = status == "active" and current["status"] == "triggered"
        row = conn.execute("update alerts set threshold = %s, note = %s, status = %s"
                           + (", triggered_at = null" if rearm else "") + " where id = %s returning *",
                           (threshold, note, status, alert_id)).fetchone()
    return _row(row)


@router.delete("/{alert_id}", status_code=204)
def delete_alert(alert_id: UUID, user: Principal = Depends(limit("write"))):
    with db.user_tx(user.user_id) as conn:
        if conn.execute("delete from alerts where id = %s", (alert_id,)).rowcount == 0:
            raise NotFound("Alert not found.")


@router.get("/firings")
def firings(user: Principal = Depends(limit("standard")),
            limit_: Annotated[int, Query(alias="limit", ge=1, le=100)] = 20):
    """Recent firings, newest first, and how many are unseen (the bell count)."""
    evaluate(user.user_id)
    with db.user_tx(user.user_id) as conn:
        rows = conn.execute(
            "select f.id, f.alert_id, f.fired_at, f.price, f.change_pct, f.seen, a.symbol, a.kind, a.threshold "
            "from alert_firings f join alerts a on a.id = f.alert_id order by f.fired_at desc limit %s",
            (limit_,)).fetchall()
        unseen = conn.execute("select count(*) as n from alert_firings where not seen").fetchone()["n"]
    return {"firings": [{"id": str(r["id"]), "alert_id": str(r["alert_id"]), "symbol": r["symbol"],
                         "kind": r["kind"], "threshold": _plain(r["threshold"]),
                         "price": _plain(r["price"]),
                         "change_pct": str(r["change_pct"]) if r["change_pct"] is not None else None,
                         "fired_at": r["fired_at"], "seen": r["seen"]} for r in rows],
            "unseen": unseen, "note": CHECK_NOTE}


@router.post("/firings/seen", status_code=204)
def mark_seen(user: Principal = Depends(limit("write"))):
    with db.user_tx(user.user_id) as conn:
        conn.execute("update alert_firings set seen = true where not seen")
