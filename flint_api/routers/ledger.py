"""Paper trading ledger: accounts, market orders, positions, activity.

Rules that protect the record:
- The server prices every order from a live quote; the client never sends a price.
- Stale quotes are refused rather than filled.
- The account row is locked while an order is checked and written, so two
  concurrent sells cannot oversell and two buys cannot overspend.
- client_order_id makes a double-submitted order land once.
- Fills are append-only; positions and cash are derived from them.
"""
import math
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, StringConstraints

import data_source

from .. import db
from ..errors import AppError, NotFound, Unavailable
from ..ratelimit import limit
from ..security import Principal
from ..validators import Symbol

router = APIRouter(prefix="/accounts", tags=["ledger"])

# Exchanges that quote in minor units: (major currency, multiplier).
_MINOR_UNITS = {"GBp": ("GBP", Decimal("0.01")), "GBX": ("GBP", Decimal("0.01")),
                "ZAc": ("ZAR", Decimal("0.01")), "ILA": ("ILS", Decimal("0.01"))}
_MAX_QUOTED_POSITIONS = 50


def _live_price(symbol: str) -> tuple[Decimal, str]:
    try:
        q = data_source.get_quote(symbol)
    except Exception:
        q = None
    price = (q or {}).get("regularMarketPrice")
    if not q or q.get("error") or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
        raise Unavailable("We couldn't get a live price for that symbol. Try again shortly.")
    if q.get("stale"):
        raise Unavailable("The latest price is out of date, so the order wasn't filled. Try again shortly.")
    currency = q.get("currency") or "USD"
    value = Decimal(str(price))
    if currency in _MINOR_UNITS:
        currency, factor = _MINOR_UNITS[currency]
        value *= factor
    return value, currency.upper()


def _fx(from_ccy: str, to_ccy: str) -> Decimal:
    """Units of to_ccy per one unit of from_ccy, from a live FX quote."""
    if from_ccy == to_ccy:
        return Decimal(1)
    rate, _ = _live_price(f"{from_ccy}{to_ccy}=X")
    return rate


def _account(conn, account_id: UUID, *, lock: bool = False) -> dict:
    sql = "select id, name, currency, starting_cash, created_at from accounts where id = %s"
    row = conn.execute(sql + (" for update" if lock else ""), (account_id,)).fetchone()
    if row is None:
        raise NotFound("Account not found.")
    return row


def _book(conn, account: dict) -> tuple[Decimal, dict]:
    """Replay fills (average-cost method). Returns cash and open positions,
    all money in the account currency."""
    cash = Decimal(account["starting_cash"])
    book: dict = {}
    fills = conn.execute(
        "select symbol, side, quantity, price, currency, fx_rate, fee from fills "
        "where account_id = %s order by filled_at, id", (account["id"],)).fetchall()
    for f in fills:
        p = book.setdefault(f["symbol"], {"quantity": Decimal(0), "cost": Decimal(0),
                                          "realized": Decimal(0), "currency": f["currency"]})
        gross = f["quantity"] * f["price"] * f["fx_rate"]
        fee = f["fee"] * f["fx_rate"]
        if f["side"] == "buy":
            cash -= gross + fee
            p["quantity"] += f["quantity"]
            p["cost"] += gross + fee
        else:
            avg = p["cost"] / p["quantity"] if p["quantity"] else Decimal(0)
            cost_out = avg * f["quantity"]
            p["realized"] += gross - fee - cost_out
            p["cost"] -= cost_out
            p["quantity"] -= f["quantity"]
            cash += gross - fee
    return cash, book


def _money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.0001")))


# ------------------------------------------------------------------ routes --

@router.get("")
def list_accounts(user: Principal = Depends(limit("standard"))):
    with db.user_tx(user.user_id) as conn:
        rows = conn.execute("select id, name, currency, starting_cash, created_at from accounts "
                            "order by created_at").fetchall()
    return {"accounts": [{**r, "id": str(r["id"]), "starting_cash": _money(r["starting_cash"])} for r in rows]}


@router.get("/{account_id}/positions")
def positions(account_id: UUID, user: Principal = Depends(limit("standard"))):
    with db.user_tx(user.user_id) as conn:
        account = _account(conn, account_id)
        cash, book = _book(conn, account)

    base = account["currency"]
    open_positions, realized_total = [], Decimal(0)
    for symbol, p in sorted(book.items()):
        realized_total += p["realized"]
        if p["quantity"] <= 0:
            continue
        row = {"symbol": symbol, "quantity": str(p["quantity"].normalize()), "currency": p["currency"],
               "avg_cost": _money(p["cost"] / p["quantity"]), "cost_basis": _money(p["cost"]),
               "realized_pnl": _money(p["realized"]),
               "last_price": None, "market_value": None, "unrealized_pnl": None, "price_unavailable": False}
        if len(open_positions) < _MAX_QUOTED_POSITIONS:
            try:
                price, ccy = _live_price(symbol)
                value = p["quantity"] * price * _fx(ccy, base)
                row.update(last_price=str(price), market_value=_money(value),
                           unrealized_pnl=_money(value - p["cost"]))
            except Unavailable:
                row["price_unavailable"] = True
        open_positions.append(row)

    return {"account": {"id": str(account["id"]), "name": account["name"], "currency": base,
                        "starting_cash": _money(account["starting_cash"])},
            "cash": _money(cash), "realized_pnl": _money(realized_total),
            "positions": open_positions}


@router.get("/{account_id}/fills")
def fills(account_id: UUID, user: Principal = Depends(limit("standard")),
          limit_: Annotated[int, Query(alias="limit", ge=1, le=200)] = 50):
    with db.user_tx(user.user_id) as conn:
        _account(conn, account_id)
        rows = conn.execute(
            "select id, symbol, side, quantity, price, currency, fx_rate, fee, stop_price, note, filled_at "
            "from fills where account_id = %s order by filled_at desc limit %s",
            (account_id, limit_)).fetchall()
    return {"fills": [{**r, "id": str(r["id"]), **{k: str(r[k]) for k in ("quantity", "price", "fx_rate", "fee")},
                       "stop_price": str(r["stop_price"]) if r["stop_price"] is not None else None}
                      for r in rows]}


class OrderBody(BaseModel):
    symbol: Symbol
    side: Literal["buy", "sell"]
    quantity: Annotated[Decimal, Field(gt=0, le=Decimal("1e9"), max_digits=20, decimal_places=8)]
    stop_price: Annotated[Decimal | None, Field(gt=0, le=Decimal("1e12"), max_digits=24, decimal_places=8)] = None
    note: Annotated[str | None, StringConstraints(max_length=500)] = None
    client_order_id: UUID


@router.post("/{account_id}/orders", status_code=201)
def place_order(account_id: UUID, body: OrderBody, user: Principal = Depends(limit("write"))):
    """Paper market order, filled at the live price."""
    with db.user_tx(user.user_id) as conn:
        base = _account(conn, account_id)["currency"]

    # Network calls happen outside the locking transaction.
    price, ccy = _live_price(body.symbol)
    fx = _fx(ccy, base)
    if body.side == "buy" and body.stop_price is not None and body.stop_price >= price:
        raise AppError("A stop for a buy must sit below the current price.")

    with db.user_tx(user.user_id) as conn:
        account = _account(conn, account_id, lock=True)
        existing = conn.execute("select id from fills where client_order_id = %s",
                                (body.client_order_id,)).fetchone()
        if existing:
            return {"fill_id": str(existing["id"]), "duplicate": True}

        cash, book = _book(conn, account)
        if body.side == "buy":
            needed = body.quantity * price * fx
            if needed > cash:
                raise AppError(f"Not enough paper cash. This order needs {_money(needed)} {base}; "
                               f"you have {_money(cash)} {base}.")
        else:
            held = book.get(body.symbol, {}).get("quantity", Decimal(0))
            if body.quantity > held:
                raise AppError(f"You can sell up to {held.normalize()} of {body.symbol}. "
                               "Short selling isn't available yet.")

        row = conn.execute(
            "insert into fills (user_id, account_id, symbol, side, quantity, price, currency, fx_rate, "
            "stop_price, note, client_order_id) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "returning id, filled_at",
            (user.user_id, account_id, body.symbol, body.side, body.quantity, price, ccy, fx,
             body.stop_price, body.note, body.client_order_id)).fetchone()

    return {"fill_id": str(row["id"]), "duplicate": False, "symbol": body.symbol, "side": body.side,
            "quantity": str(body.quantity), "price": str(price), "currency": ccy, "fx_rate": str(fx),
            "filled_at": row["filled_at"]}
