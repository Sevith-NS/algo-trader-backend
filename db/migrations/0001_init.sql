-- Flint OS v2: initial schema.
--
-- Isolation model: every user-owned row carries user_id (the Clerk user id).
-- Row-level security is ENABLED and FORCED on every table, and every policy
-- compares against app_user_id(), which reads a transaction-local setting the
-- API sets from the verified session token. The API connects as flint_app,
-- which cannot bypass RLS, cannot run DDL, and cannot change its own plan.

create or replace function app_user_id() returns text
  language sql stable
  as $$ select nullif(current_setting('app.user_id', true), '') $$;

-- ------------------------------------------------------------------ users --
create table users (
  id                  text primary key check (id ~ '^user_[A-Za-z0-9]{1,64}$'),
  created_at          timestamptz not null default now(),
  -- Entitlement tier. Changed only by billing/admin, never by the API role.
  plan                text not null default 'free' check (plan in ('free', 'pro')),
  base_currency       char(3) not null default 'INR' check (base_currency ~ '^[A-Z]{3}$'),
  risk_per_trade_pct  numeric(5, 2) not null default 1.00
                        check (risk_per_trade_pct > 0 and risk_per_trade_pct <= 10),
  markets             text[] not null default '{}' check (cardinality(markets) <= 16),
  onboarded_at        timestamptz
);

-- --------------------------------------------------------- paper accounts --
create table accounts (
  id             uuid primary key default gen_random_uuid(),
  user_id        text not null references users (id) on delete cascade,
  name           text not null default 'Paper' check (length(name) between 1 and 40),
  currency       char(3) not null check (currency ~ '^[A-Z]{3}$'),
  starting_cash  numeric(20, 4) not null check (starting_cash > 0 and starting_cash <= 1e12),
  created_at     timestamptz not null default now(),
  unique (user_id, name),
  unique (id, user_id)
);

-- Append-only ledger of paper executions. Positions and P&L derive from it.
create table fills (
  id          uuid primary key default gen_random_uuid(),
  user_id     text not null,
  account_id  uuid not null,
  symbol      text not null check (symbol ~ '^[A-Z0-9^][A-Z0-9.=&^-]{0,23}$'),
  side        text not null check (side in ('buy', 'sell')),
  quantity    numeric(24, 8) not null check (quantity > 0 and quantity <= 1e12),
  price       numeric(24, 8) not null check (price > 0 and price <= 1e12),
  currency    char(3) not null check (currency ~ '^[A-Z]{3}$'),
  fee         numeric(20, 8) not null default 0 check (fee >= 0),
  stop_price  numeric(24, 8) check (stop_price is null or stop_price > 0),
  note        text check (note is null or length(note) <= 500),
  filled_at   timestamptz not null default now(),
  -- The account must belong to the same user as the fill.
  foreign key (account_id, user_id) references accounts (id, user_id) on delete cascade
);
create index fills_account_time on fills (account_id, filled_at);

-- -------------------------------------------------------------- watchlists --
create table watchlists (
  id          uuid primary key default gen_random_uuid(),
  user_id     text not null references users (id) on delete cascade,
  name        text not null check (length(name) between 1 and 40),
  created_at  timestamptz not null default now(),
  unique (user_id, name),
  unique (id, user_id)
);

create table watchlist_items (
  watchlist_id  uuid not null,
  user_id       text not null,
  symbol        text not null check (symbol ~ '^[A-Z0-9^][A-Z0-9.=&^-]{0,23}$'),
  added_at      timestamptz not null default now(),
  primary key (watchlist_id, symbol),
  foreign key (watchlist_id, user_id) references watchlists (id, user_id) on delete cascade
);

-- --------------------------------------------------------- decision journal --
-- The AI desk's past decisions. Soft delete with a restore window; a purge
-- job run by the owner role removes rows deleted more than 30 days ago.
create table journal_entries (
  id          uuid primary key default gen_random_uuid(),
  user_id     text not null references users (id) on delete cascade,
  symbol      text not null check (symbol ~ '^[A-Z0-9^][A-Z0-9.=&^-]{0,23}$'),
  created_at  timestamptz not null default now(),
  deleted_at  timestamptz,
  entry       jsonb not null check (pg_column_size(entry) <= 65536),
  outcome     jsonb check (outcome is null or pg_column_size(outcome) <= 8192)
);
create index journal_live on journal_entries (user_id, created_at desc) where deleted_at is null;

-- ---------------------------------------------------------- row-level security --
alter table users           enable row level security;
alter table accounts        enable row level security;
alter table fills           enable row level security;
alter table watchlists      enable row level security;
alter table watchlist_items enable row level security;
alter table journal_entries enable row level security;

alter table users           force row level security;
alter table accounts        force row level security;
alter table fills           force row level security;
alter table watchlists      force row level security;
alter table watchlist_items force row level security;
alter table journal_entries force row level security;

create policy own_row on users
  using (id = app_user_id()) with check (id = app_user_id());
create policy own_rows on accounts
  using (user_id = app_user_id()) with check (user_id = app_user_id());
create policy own_rows on fills
  using (user_id = app_user_id()) with check (user_id = app_user_id());
create policy own_rows on watchlists
  using (user_id = app_user_id()) with check (user_id = app_user_id());
create policy own_rows on watchlist_items
  using (user_id = app_user_id()) with check (user_id = app_user_id());
create policy own_rows on journal_entries
  using (user_id = app_user_id()) with check (user_id = app_user_id());

-- -------------------------------------------------------- least-privilege role --
-- flint_app is created by the migration runner (it owns the password). It gets
-- data access only: no DDL, no TRUNCATE, and no BYPASSRLS.
revoke all on all tables in schema public from public;
grant usage on schema public to flint_app;
grant select, insert, delete on users to flint_app;
grant update (base_currency, risk_per_trade_pct, markets, onboarded_at) on users to flint_app;
grant select, insert, update on accounts to flint_app;
grant select, insert on fills to flint_app;                       -- ledger is append-only
grant select, insert, update, delete on watchlists to flint_app;
grant select, insert, delete on watchlist_items to flint_app;
grant select, insert on journal_entries to flint_app;
grant update (deleted_at, outcome) on journal_entries to flint_app; -- soft delete, scoring
grant execute on function app_user_id() to flint_app;
