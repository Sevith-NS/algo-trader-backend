"""Storage adapters for engines that keep state: per user, plus shared research."""
import json
import uuid

from . import db

_JOURNAL_MAX = 500


class DbJournalStore:
    """The desk's notes for one user, in Postgres.

    Same two methods as agents_desk.FileJournalStore. Each call is its own
    short transaction, because the desk streams for ~40s and must not hold a
    connection open while it does.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id

    def entries(self) -> list:
        with db.user_tx(self.user_id) as conn:
            rows = conn.execute(
                "select id, entry, outcome from journal_entries where deleted_at is null "
                "order by created_at desc limit %s", (_JOURNAL_MAX,)).fetchall()
        out = []
        for r in reversed(rows):  # oldest first, as the desk expects
            e = dict(r["entry"])
            e["id"] = str(r["id"])
            e["outcome"] = r["outcome"]
            out.append(e)
        return out

    def append(self, entry: dict) -> dict:
        row_id = uuid.uuid4()
        entry["id"] = str(row_id)
        body = {k: v for k, v in entry.items() if k not in ("id", "outcome")}
        with db.user_tx(self.user_id) as conn:
            conn.execute(
                "insert into journal_entries (id, user_id, symbol, entry) values (%s, %s, %s, %s::jsonb)",
                (row_id, self.user_id, entry["symbol"], json.dumps(body, default=str)))
        return entry


# The research day is India's: one desk run or teardown per symbol per IST date.
_TODAY = "(now() at time zone 'Asia/Kolkata')::date"


class DbResearchRuns:
    """Model output about a symbol (desk runs, deep-dive teardowns), shared by
    every user because no user's data goes into it. No user_tx: the rows
    belong to no one, so there is nothing for row-level security to filter."""

    def latest(self, kind: str, symbol: str, variant: str = "") -> dict | None:
        """{"payload", "day", "today"} for the newest row, or None."""
        with db.pool().connection() as conn:
            return conn.execute(
                f"select payload, day::text as day, day = {_TODAY} as today from research_runs "
                "where kind = %s and symbol = %s and variant = %s order by day desc limit 1",
                (kind, symbol, variant)).fetchone()

    def put(self, kind: str, symbol: str, variant: str, payload: dict) -> None:
        """First write of the day wins; a concurrent duplicate run is dropped."""
        with db.pool().connection() as conn:
            conn.execute(
                "insert into research_runs (kind, symbol, variant, day, payload) "
                f"values (%s, %s, %s, {_TODAY}, %s::jsonb) on conflict do nothing",
                (kind, symbol, variant, json.dumps(payload, default=str)))
