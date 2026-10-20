"""Accounts: the paper trading ledger (orders, positions, activity) and the
read-only real accounts of the user's broker connections (holdings, live
from the broker), plus performance.

Rules that protect the record:
- The server prices every order from a live quote; the client never sends a price.
- Stale quotes are refused rather than filled.
- The account row is locked while an order is checked and written, so two
  concurrent sells cannot oversell and two buys cannot overspend.
- client_order_id makes a double-submitted order land once.
- Fills are append-only; positions and cash are derived from them.
- Real accounts never take orders: they aren't rows in the database at all,
  only broker connections held in the user's browser (broker_jar.py).
"""
import math
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel, Field, StringConstraints

import data_source

from .. import db, performance
from ..broker_jar import Jar
from ..brokers import ADAPTERS
from ..errors import AppError, Forbidden, NotFound, Unavailable
from .brokers import label as broker_label
from .brokers import live as live_broker
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
    sql = "select id, name, currency, starting_cash, created_at, kind from accounts where id = %s"
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
def list_accounts(request: Request, user: Principal = Depends(limit("standard"))):
    """Paper accounts from the database, then one real account per broker
    connection in this browser (its id is the connection id)."""
    with db.user_tx(user.user_id) as conn:
        rows = conn.execute("select id, name, currency, starting_cash, created_at, kind from accounts "
                            "order by created_at").fetchall()
    paper = [{"id": str(r["id"]), "name": r["name"], "kind": r["kind"], "currency": r["currency"],
              "starting_cash": _money(r["starting_cash"]), "created_at": r["created_at"], "broker": None, "sync": None}
             for r in rows]
    real = [{"id": c.id, "name": f"{broker_label(c)} {c.broker_user_id}"[:40], "kind": "real",
             "currency": ADAPTERS[c.broker].CURRENCY,
             "starting_cash": None, "created_at": c.connected_at, "broker": c.broker,
             "sync": {"connection_id": c.id, "status": c.status, "last_synced_at": None}}
            for c in Jar.of(request, user.user_id).connections()]
    return {"accounts": paper + real}


def _paper(account: dict) -> dict:
    if account["kind"] != "paper":
        raise Forbidden("Real accounts are read-only. Paper orders and positions live on paper accounts.")
    return account


@router.get("/{account_id}/positions")
def positions(account_id: UUID, user: Principal = Depends(limit("standard"))):
    with db.user_tx(user.user_id) as conn:
        account = _paper(_account(conn, account_id))
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
               "last_price": None, "fx_rate": None, "market_value": None, "unrealized_pnl": None, "price_unavailable": False}
        if len(open_positions) < _MAX_QUOTED_POSITIONS:
            try:
                price, ccy = _live_price(symbol)
                rate = _fx(ccy, base)
                value = p["quantity"] * price * rate
                row.update(last_price=str(price), fx_rate=str(rate), market_value=_money(value),
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
        base = _paper(_account(conn, account_id))["currency"]

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


# ---------------------------------------------------------- real holdings --

def _pct(num: Decimal, den: Decimal) -> str | None:
    return str((num / den * 100).quantize(Decimal("0.01"))) if den else None


@router.get("/{account_id}/holdings")
def holdings(account_id: UUID, request: Request, response: Response, user: Principal = Depends(limit("expensive"))):
    """A real account's holdings, live from the broker. Each row is in its own
    currency; `market_value_base` converts it to the user's base currency."""
    jar = Jar.of(request, user.user_id)
    c = jar.get(str(account_id))
    with db.user_tx(user.user_id) as conn:
        if c is None:
            _account(conn, account_id)
            raise AppError("Paper accounts show positions, not broker holdings.")
        base = conn.execute("select base_currency from users").fetchone()["base_currency"]
    rows = sorted(live_broker(jar, c, positions=False)[0], key=lambda r: r["symbol"])
    response.headers.update(jar.headers())
    as_of = datetime.now(timezone.utc)
    account = {"id": c.id, "name": f"{broker_label(c)} {c.broker_user_id}"[:40],
               "currency": ADAPTERS[c.broker].CURRENCY}

    rates: dict[str, Decimal | None] = {}
    for ccy in {r["currency"] for r in rows}:
        try:
            rates[ccy] = _fx(ccy, base)
        except Unavailable:
            rates[ccy] = None

    # Cost is unknown for some brokers (crypto balances), and a price can be
    # missing; those parts are null and left out of totals rather than guessed.
    out, total_value, total_invested, total_pnl, total_base = [], Decimal(0), Decimal(0), Decimal(0), Decimal(0)
    base_complete = True
    for r in rows:
        price = r["last_price"] or r["close_price"] or r["avg_price"]
        value = r["quantity"] * price if price is not None else None
        invested = r["quantity"] * r["avg_price"] if r["avg_price"] is not None else None
        pnl = value - invested if value is not None and invested is not None else None
        rate = rates.get(r["currency"])
        if value is not None:
            total_value += value
        if pnl is not None:
            total_invested += invested
            total_pnl += pnl
        if rate is None or value is None:
            base_complete = False
        else:
            total_base += value * rate
        day = (_pct(r["last_price"] - r["close_price"], r["close_price"])
               if r["last_price"] is not None and r["close_price"] else None)
        out.append({"symbol": r["symbol"], "broker_symbol": r["broker_symbol"], "exchange": r["exchange"],
                    "isin": r["isin"], "quantity": format(r["quantity"].normalize(), "f"), "currency": r["currency"],
                    "avg_price": _money(r["avg_price"]) if r["avg_price"] is not None else None,
                    "last_price": _money(r["last_price"]) if r["last_price"] is not None else None,
                    "market_value": _money(value) if value is not None else None,
                    "invested": _money(invested) if invested is not None else None,
                    "unrealized_pnl": _money(pnl) if pnl is not None else None,
                    "unrealized_pnl_pct": _pct(pnl, invested) if pnl is not None else None,
                    "day_change_pct": day,
                    "fx_rate": str(rate) if rate is not None else None,
                    "market_value_base": _money(value * rate) if rate is not None and value is not None else None})

    return {"account": {"id": str(account["id"]), "name": account["name"], "kind": "real",
                        "currency": account["currency"]},
            "as_of": as_of,
            "base_currency": base,
            "holdings": out,
            "totals": {"market_value": _money(total_value), "invested": _money(total_invested),
                       "unrealized_pnl": _money(total_pnl),
                       "unrealized_pnl_pct": _pct(total_pnl, total_invested),
                       "market_value_base": _money(total_base) if base_complete else None}}


# ------------------------------------------------------------- performance --

@router.get("/{account_id}/performance")
def account_performance(account_id: UUID, request: Request, user: Principal = Depends(limit("expensive")),
                        period: Literal["1m", "3m", "6m", "1y", "ytd", "all"] = "6m"):
    if Jar.of(request, user.user_id).get(str(account_id)) is not None:
        # Broker data is never stored, so there is no history to measure.
        return {"account_id": str(account_id), "kind": "real", "currency": "INR", "period": period,
                "start": None, "end": None, "twr_pct": None, "xirr_pct": None, "series": [], "benchmark": None,
                "method": "Broker accounts are read live and nothing is stored, so there is no history to chart.",
                "xirr_note": "Performance over time is only kept for paper accounts."}
    today = date.today()
    with db.user_tx(user.user_id) as conn:
        account = _account(conn, account_id)
        start = performance.period_start(period, account["created_at"].date(), today)
        fills_ = conn.execute("select symbol, side, quantity, price, currency, fx_rate, fee, filled_at "
                              "from fills where account_id = %s", (account_id,)).fetchall()

    dates, values = performance.paper_values(account, fills_, start, today)
    flows = [0.0] * len(values)
    method = "Paper: the fill ledger replayed against daily closes. No deposits or withdrawals."

    result = {"account_id": str(account_id), "kind": account["kind"], "currency": account["currency"],
              "period": period, "start": dates[0].isoformat() if dates else None,
              "end": dates[-1].isoformat() if dates else None, "method": method}
    if len(dates) < 2:
        return {**result, "twr_pct": None, "xirr_pct": None,
                "xirr_note": "Not enough history yet.", "series": [], "benchmark": None}
    return {**result, **performance.summarise(dates, values, flows),
            "benchmark": performance.benchmark(account["currency"], dates, today)}
