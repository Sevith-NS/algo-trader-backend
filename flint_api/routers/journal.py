"""The decision journal: the user's own scored AI-desk decisions.

Deleting is a soft delete with a 30-day restore window. This replaces the old
unauthenticated hard delete of a shared journal (audit finding 1).
"""
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query

import agents_desk

from .. import db
from ..errors import NotFound
from ..ratelimit import limit
from ..security import Principal
from ..stores import DbJournalStore
from ..validators import Symbol

router = APIRouter(prefix="/journal", tags=["journal"])

RESTORE_DAYS = 30


@router.get("")
def list_entries(user: Principal = Depends(limit("standard")),
                 symbol: Symbol | None = None,
                 limit_: Annotated[int, Query(alias="limit", ge=1, le=200)] = 60):
    return agents_desk.get_journal(symbol, limit=limit_, store=DbJournalStore(user.user_id))


@router.delete("/{entry_id}")
def delete_entry(entry_id: UUID, user: Principal = Depends(limit("write"))):
    with db.user_tx(user.user_id) as conn:
        row = conn.execute(
            "update journal_entries set deleted_at = now() where id = %s and deleted_at is null "
            "returning deleted_at + make_interval(days => %s) as restore_until",
            (entry_id, RESTORE_DAYS)).fetchone()
    if row is None:
        raise NotFound("No journal entry with that id.")
    return {"deleted": str(entry_id), "restore_until": row["restore_until"]}


@router.post("/{entry_id}/restore")
def restore_entry(entry_id: UUID, user: Principal = Depends(limit("write"))):
    with db.user_tx(user.user_id) as conn:
        row = conn.execute(
            "update journal_entries set deleted_at = null where id = %s and deleted_at is not null "
            "and deleted_at > now() - make_interval(days => %s) returning id",
            (entry_id, RESTORE_DAYS)).fetchone()
    if row is None:
        raise NotFound("Nothing to restore. The entry doesn't exist or its restore window has passed.")
    return {"restored": str(entry_id)}
