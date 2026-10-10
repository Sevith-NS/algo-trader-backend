"""Postgres access scoped to one user.

All user data goes through `user_tx(user_id)`. It opens a transaction as the
least-privilege flint_app role and sets `app.user_id` for that transaction
only, so row-level security filters every statement to the caller's rows.
The id comes from the verified Clerk token, never from the request body.
"""
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .errors import Unavailable
from .settings import settings

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        if not settings.database_url_app:
            raise Unavailable("Your data is unavailable right now.")
        _pool = ConnectionPool(
            settings.database_url_app,
            min_size=0, max_size=10, open=True,
            # Neon's pooler runs in transaction mode: no server-side prepares.
            kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
        )
    return _pool


@contextmanager
def user_tx(user_id: str) -> Iterator[psycopg.Connection]:
    with pool().connection() as conn:
        with conn.transaction():
            conn.execute("select set_config('app.user_id', %s, true)", (user_id,))
            yield conn


def ensure_user(conn: psycopg.Connection, user_id: str) -> None:
    """First authenticated request creates the user's row; later ones are no-ops."""
    conn.execute("insert into users (id) values (%s) on conflict (id) do nothing", (user_id,))


def close() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
