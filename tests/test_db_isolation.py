"""Row-level security, proven against the real Neon dev branch.

Runs as the least-privilege flint_app role, exactly as the API does.
Skipped when DATABASE_URL_APP is not configured.
"""
import uuid

import psycopg
import pytest

from flint_api import db
from flint_api.settings import settings

pytestmark = pytest.mark.skipif(not settings.database_url_app, reason="no DATABASE_URL_APP")


def _uid() -> str:
    return "user_t" + uuid.uuid4().hex[:16]


@pytest.fixture()
def two_users():
    a, b = _uid(), _uid()
    for u in (a, b):
        with db.user_tx(u) as conn:
            db.ensure_user(conn, u)
    yield a, b
    for u in (a, b):
        with db.user_tx(u) as conn:
            conn.execute("delete from users where id = %s", (u,))


def _watchlist(user: str, name: str = "Core") -> str:
    with db.user_tx(user) as conn:
        row = conn.execute("insert into watchlists (user_id, name) values (%s, %s) returning id",
                           (user, name)).fetchone()
        conn.execute("insert into watchlist_items (watchlist_id, user_id, symbol) values (%s, %s, 'AAPL')",
                     (row["id"], user))
    return row["id"]


def test_users_see_only_their_own_rows(two_users):
    a, b = two_users
    wl = _watchlist(a)
    with db.user_tx(b) as conn:
        assert conn.execute("select * from watchlists").fetchall() == []
        assert conn.execute("select * from watchlist_items").fetchall() == []
        assert conn.execute("select * from users").fetchall() == [{**conn.execute(
            "select * from users where id = %s", (b,)).fetchone()}]
        # Reaching for A's row by id finds nothing...
        assert conn.execute("select * from watchlists where id = %s", (wl,)).fetchone() is None
        # ...and writes against it affect nothing.
        assert conn.execute("update watchlists set name = 'pwned' where id = %s", (wl,)).rowcount == 0
        assert conn.execute("delete from watchlists where id = %s", (wl,)).rowcount == 0
    with db.user_tx(a) as conn:
        assert conn.execute("select name from watchlists where id = %s", (wl,)).fetchone()["name"] == "Core"


def test_cannot_write_rows_owned_by_someone_else(two_users):
    a, b = two_users
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with db.user_tx(b) as conn:
            conn.execute("insert into watchlists (user_id, name) values (%s, 'Sneaky')", (a,))


def test_cannot_attach_items_to_another_users_list(two_users):
    a, b = two_users
    wl = _watchlist(a)
    with pytest.raises(psycopg.Error):
        with db.user_tx(b) as conn:
            conn.execute("insert into watchlist_items (watchlist_id, user_id, symbol) values (%s, %s, 'TSLA')",
                         (wl, b))


def test_no_user_context_sees_nothing(two_users):
    a, _ = two_users
    _watchlist(a)
    with db.pool().connection() as conn:
        assert conn.execute("select count(*) as n from watchlists").fetchone()["n"] == 0
        assert conn.execute("select count(*) as n from users").fetchone()["n"] == 0


def test_user_context_does_not_leak_across_transactions(two_users):
    a, _ = two_users
    _watchlist(a)
    with db.user_tx(a):
        pass
    with db.pool().connection() as conn:
        assert conn.execute("select current_setting('app.user_id', true) as v").fetchone()["v"] in (None, "")


def test_app_role_cannot_upgrade_its_own_plan(two_users):
    a, _ = two_users
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with db.user_tx(a) as conn:
            conn.execute("update users set plan = 'pro' where id = %s", (a,))


def test_ledger_is_append_only(two_users):
    a, _ = two_users
    with db.user_tx(a) as conn:
        acct = conn.execute("insert into accounts (user_id, currency, starting_cash) values (%s, 'INR', 1000000) "
                            "returning id", (a,)).fetchone()["id"]
        conn.execute("insert into fills (user_id, account_id, symbol, side, quantity, price, currency) "
                     "values (%s, %s, 'RELIANCE.NS', 'buy', 10, 2900, 'INR')", (a, acct))
    for stmt in ("update fills set price = 1", "delete from fills"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with db.user_tx(a) as conn:
                conn.execute(stmt)


def test_app_role_cannot_change_schema_or_bypass_rls(two_users):
    a, _ = two_users
    for stmt in ("alter table users disable row level security",
                 "drop table watchlists",
                 "truncate fills",
                 "set role neondb_owner"):
        with pytest.raises(psycopg.Error):
            with db.user_tx(a) as conn:
                conn.execute(stmt)
    with db.pool().connection() as conn:
        role = conn.execute("select rolbypassrls, rolsuper from pg_roles where rolname = current_user").fetchone()
        assert role == {"rolbypassrls": False, "rolsuper": False}


def test_deleting_a_user_removes_all_their_data(two_users):
    a, _ = two_users
    _watchlist(a)
    with db.user_tx(a) as conn:
        conn.execute("delete from users where id = %s", (a,))
    with db.user_tx(a) as conn:
        assert conn.execute("select count(*) as n from watchlists").fetchone()["n"] == 0
        assert conn.execute("select count(*) as n from watchlist_items").fetchone()["n"] == 0
