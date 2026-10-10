"""Watchlists. Every query runs under the caller's row-level security scope,
so an id belonging to someone else behaves exactly like one that doesn't exist."""
from typing import Annotated
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends
from pydantic import BaseModel, StringConstraints

from .. import db
from ..errors import AppError, Conflict, NotFound
from ..ratelimit import limit
from ..security import Principal
from ..validators import Symbol

router = APIRouter(prefix="/watchlists", tags=["watchlists"])

MAX_LISTS = 20
MAX_ITEMS = 100

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]


class ListBody(BaseModel):
    name: Name


class ItemBody(BaseModel):
    symbol: Symbol


def _lists(conn) -> list:
    rows = conn.execute(
        "select w.id, w.name, w.created_at, "
        "coalesce(array_agg(i.symbol order by i.added_at) filter (where i.symbol is not null), '{}') as symbols "
        "from watchlists w left join watchlist_items i on i.watchlist_id = w.id "
        "group by w.id order by w.created_at").fetchall()
    return [{**r, "id": str(r["id"])} for r in rows]


@router.get("")
def list_watchlists(user: Principal = Depends(limit("standard"))):
    with db.user_tx(user.user_id) as conn:
        return {"watchlists": _lists(conn)}


@router.post("", status_code=201)
def create_watchlist(body: ListBody, user: Principal = Depends(limit("write"))):
    try:
        with db.user_tx(user.user_id) as conn:
            if conn.execute("select count(*) as n from watchlists").fetchone()["n"] >= MAX_LISTS:
                raise AppError(f"You can have up to {MAX_LISTS} watchlists.")
            row = conn.execute("insert into watchlists (user_id, name) values (%s, %s) returning id",
                               (user.user_id, body.name)).fetchone()
    except psycopg.errors.UniqueViolation:
        raise Conflict("You already have a watchlist with that name.")
    return {"id": str(row["id"]), "name": body.name, "symbols": []}


@router.patch("/{list_id}")
def rename_watchlist(list_id: UUID, body: ListBody, user: Principal = Depends(limit("write"))):
    try:
        with db.user_tx(user.user_id) as conn:
            row = conn.execute("update watchlists set name = %s where id = %s returning id",
                               (body.name, list_id)).fetchone()
    except psycopg.errors.UniqueViolation:
        raise Conflict("You already have a watchlist with that name.")
    if row is None:
        raise NotFound("Watchlist not found.")
    return {"id": str(list_id), "name": body.name}


@router.delete("/{list_id}", status_code=204)
def delete_watchlist(list_id: UUID, user: Principal = Depends(limit("write"))):
    with db.user_tx(user.user_id) as conn:
        if conn.execute("delete from watchlists where id = %s", (list_id,)).rowcount == 0:
            raise NotFound("Watchlist not found.")


@router.post("/{list_id}/items", status_code=201)
def add_item(list_id: UUID, body: ItemBody, user: Principal = Depends(limit("write"))):
    with db.user_tx(user.user_id) as conn:
        if conn.execute("select 1 from watchlists where id = %s", (list_id,)).fetchone() is None:
            raise NotFound("Watchlist not found.")
        count = conn.execute("select count(*) as n from watchlist_items where watchlist_id = %s",
                             (list_id,)).fetchone()["n"]
        if count >= MAX_ITEMS:
            raise AppError(f"A watchlist holds up to {MAX_ITEMS} symbols.")
        conn.execute("insert into watchlist_items (watchlist_id, user_id, symbol) values (%s, %s, %s) "
                     "on conflict do nothing", (list_id, user.user_id, body.symbol))
    return {"id": str(list_id), "symbol": body.symbol}


@router.delete("/{list_id}/items/{symbol}", status_code=204)
def remove_item(list_id: UUID, symbol: Symbol, user: Principal = Depends(limit("write"))):
    with db.user_tx(user.user_id) as conn:
        if conn.execute("delete from watchlist_items where watchlist_id = %s and symbol = %s",
                        (list_id, symbol)).rowcount == 0:
            raise NotFound("That symbol isn't on this watchlist.")
