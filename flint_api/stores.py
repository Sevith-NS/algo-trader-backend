"""Per-user storage adapters for engines that keep state."""
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
