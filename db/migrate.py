"""Apply SQL migrations to the linked Neon branch, as the owner role.

    python -m db.migrate              apply pending migrations
    python -m db.migrate --app-role   also (re)create the flint_app login role
                                      and write DATABASE_URL_APP to .env

Secrets never reach stdout: the role password is generated here, written to
.env (gitignored), and set on the role in one step.
"""
import argparse
import os
import secrets
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, quote

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "db" / "migrations"
ENV_FILE = ROOT / ".env"

sys.path.insert(0, str(ROOT))
from flint_api.settings import settings  # noqa: E402  (loads .env)


def _owner_url() -> str:
    url = settings.database_url_owner or os.environ.get("DATABASE_URL", "")
    if not url:
        sys.exit("DATABASE_URL_UNPOOLED is not set. Run `neon link` first.")
    return url


def _set_env_var(name: str, value: str) -> None:
    text = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = [ln for ln in text.splitlines() if not ln.startswith(f"{name}=")]
    lines.append(f"{name}={value}")
    ENV_FILE.write_bytes((nl.join(lines) + nl).encode("utf-8"))


def ensure_app_role(conn: psycopg.Connection) -> None:
    password = secrets.token_urlsafe(32)
    exists = conn.execute("select 1 from pg_roles where rolname = 'flint_app'").fetchone()
    verb = sql.SQL("alter role") if exists else sql.SQL("create role")
    conn.execute(sql.SQL("{} flint_app with login nobypassrls nocreatedb nocreaterole password {}")
                 .format(verb, sql.Literal(password)))
    # Same host as the pooled owner URL, different credentials.
    pooled = os.environ.get("DATABASE_URL", "") or _owner_url()
    parts = urlsplit(pooled)
    netloc = f"flint_app:{quote(password, safe='')}@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    _set_env_var("DATABASE_URL_APP", urlunsplit(parts._replace(netloc=netloc)))
    print("flint_app role ready; DATABASE_URL_APP written to .env")


def apply_migrations(conn: psycopg.Connection) -> None:
    conn.execute("""create table if not exists schema_migrations (
                      name text primary key, applied_at timestamptz not null default now())""")
    conn.execute("revoke all on schema_migrations from public")
    done = {r[0] for r in conn.execute("select name from schema_migrations")}
    pending = sorted(p for p in MIGRATIONS.glob("*.sql") if p.name not in done)
    for path in pending:
        with conn.transaction():
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("insert into schema_migrations (name) values (%s)", (path.name,))
        print(f"applied {path.name}")
    if not pending:
        print("no pending migrations")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-role", action="store_true")
    args = parser.parse_args()
    with psycopg.connect(_owner_url(), autocommit=True) as conn:
        if args.app_role:
            ensure_app_role(conn)
        apply_migrations(conn)


if __name__ == "__main__":
    main()
