-- Polaris v3: real (broker-synced, read-only) accounts next to paper ones,
-- holdings snapshots, daily values for performance, and price alerts.
-- Same isolation model as 0001: user_id on every row, RLS forced, policies on
-- app_user_id(), and flint_app gets data access only.

-- -------------------------------------------------------- broker connections --
-- One row per linked broker login. The access token is AES-GCM encrypted by
-- the API (flint_api/secretbox.py) with the row's user and id as associated
-- data, so a ciphertext copied to another row or user does not decrypt.
create table broker_connections (
  id                uuid primary key default gen_random_uuid(),
  user_id           text not null references users (id) on delete cascade,
  broker            text not null check (broker in ('zerodha')),
  broker_user_id    text not null check (length(broker_user_id) between 1 and 64),
  broker_user_name  text check (broker_user_name is null or length(broker_user_name) <= 120),
  status            text not null default 'connected'
                      check (status in ('connected', 'syncing', 'needs_reconnect', 'error')),
  access_token_enc  bytea check (access_token_enc is null or length(access_token_enc) <= 1024),
  token_expires_at  timestamptz,
  last_synced_at    timestamptz,
  last_error        text check (last_error is null or length(last_error) <= 300),
  created_at        timestamptz not null default now(),
  unique (user_id, broker, broker_user_id),
  unique (id, user_id)
);

-- A pending "connect" round trip. Only the SHA-256 of the state nonce is kept;
-- the nonce itself travels through the broker's login and back once.
create table broker_link_states (
  state_hash  bytea primary key check (length(state_hash) = 32),
  user_id     text not null references users (id) on delete cascade,
  broker      text not null check (broker in ('zerodha')),
  expires_at  timestamptz not null
);

-- ------------------------------------------------------------ account kinds --
-- Paper accounts keep a starting balance and a fill ledger. Real accounts
-- belong to a broker connection, have no starting balance and never take fills.
alter table accounts add column kind text not null default 'paper' check (kind in ('paper', 'real'));
alter table accounts add column broker_connection_id uuid;
alter table accounts alter column starting_cash drop not null;
alter table accounts drop constraint accounts_starting_cash_check;
alter table accounts add constraint accounts_kind_shape check (
  (kind = 'paper' and broker_connection_id is null
     and starting_cash is not null and starting_cash > 0 and starting_cash <= 1e12)
  or (kind = 'real' and broker_connection_id is not null and starting_cash is null));
alter table accounts add constraint accounts_broker_connection
  foreign key (broker_connection_id, user_id) references broker_connections (id, user_id) on delete cascade;
create unique index accounts_one_per_connection on accounts (broker_connection_id)
  where broker_connection_id is not null;

-- Defence in depth for "real is read-only": the API refuses orders on a real
-- account, and the database refuses the fill even if that check were skipped.
create or replace function fills_paper_only() returns trigger
  language plpgsql
  as $$
begin
  if exists (select 1 from accounts where id = new.account_id and kind <> 'paper') then
    raise exception 'fills can only be recorded on paper accounts' using errcode = 'check_violation';
  end if;
  return new;
end
$$;
create trigger fills_paper_only before insert on fills
  for each row execute function fills_paper_only();

-- ----------------------------------------------------------------- holdings --
-- The latest synced holdings of a real account, replaced on every sync.
create table holdings (
  account_id     uuid not null,
  user_id        text not null,
  symbol         text not null check (symbol ~ '^[A-Z0-9^][A-Z0-9.=&^-]{0,23}$'),
  broker_symbol  text not null check (length(broker_symbol) between 1 and 40),
  exchange       text not null check (exchange in ('NSE', 'BSE')),
  isin           text check (isin is null or isin ~ '^[A-Z0-9]{12}$'),
  quantity       numeric(24, 8) not null check (quantity >= 0),
  avg_price      numeric(24, 8) not null check (avg_price >= 0),
  last_price     numeric(24, 8) check (last_price is null or last_price >= 0),
  close_price    numeric(24, 8) check (close_price is null or close_price >= 0),
  currency       char(3) not null check (currency ~ '^[A-Z]{3}$'),
  synced_at      timestamptz not null,
  primary key (account_id, symbol),
  foreign key (account_id, user_id) references accounts (id, user_id) on delete cascade
);

-- One value per account per day, written on sync. Real accounts have no fill
-- ledger, so performance is measured from these.
create table account_snapshots (
  account_id    uuid not null,
  user_id       text not null,
  on_date       date not null,
  market_value  numeric(24, 4) not null check (market_value >= 0),
  invested      numeric(24, 4) not null check (invested >= 0),
  currency      char(3) not null check (currency ~ '^[A-Z]{3}$'),
  primary key (account_id, on_date),
  foreign key (account_id, user_id) references accounts (id, user_id) on delete cascade
);

-- ------------------------------------------------------------------- alerts --
create table alerts (
  id               uuid primary key default gen_random_uuid(),
  user_id          text not null references users (id) on delete cascade,
  symbol           text not null check (symbol ~ '^[A-Z0-9^][A-Z0-9.=&^-]{0,23}$'),
  kind             text not null check (kind in ('price_above', 'price_below', 'move_pct')),
  threshold        numeric(24, 8) not null check (threshold > 0 and threshold <= 1e12),
  status           text not null default 'active' check (status in ('active', 'paused', 'triggered')),
  note             text check (note is null or length(note) <= 200),
  reference_price  numeric(24, 8) check (reference_price is null or reference_price > 0),
  created_at       timestamptz not null default now(),
  triggered_at     timestamptz,
  last_checked_at  timestamptz,
  unique (id, user_id)
);
create index alerts_active on alerts (user_id) where status = 'active';

create table alert_firings (
  id          uuid primary key default gen_random_uuid(),
  alert_id    uuid not null,
  user_id     text not null,
  fired_at    timestamptz not null default now(),
  price       numeric(24, 8) not null check (price > 0),
  change_pct  numeric(12, 4),
  seen        boolean not null default false,
  foreign key (alert_id, user_id) references alerts (id, user_id) on delete cascade
);
create index alert_firings_recent on alert_firings (user_id, fired_at desc);

-- ------------------------------------------------------- row-level security --
alter table broker_connections enable row level security;
alter table broker_link_states enable row level security;
alter table holdings           enable row level security;
alter table account_snapshots  enable row level security;
alter table alerts             enable row level security;
alter table alert_firings      enable row level security;

alter table broker_connections force row level security;
alter table broker_link_states force row level security;
alter table holdings           force row level security;
alter table account_snapshots  force row level security;
alter table alerts             force row level security;
alter table alert_firings      force row level security;

create policy own_rows on broker_connections
  using (user_id = app_user_id()) with check (user_id = app_user_id());
create policy own_rows on broker_link_states
  using (user_id = app_user_id()) with check (user_id = app_user_id());
create policy own_rows on holdings
  using (user_id = app_user_id()) with check (user_id = app_user_id());
create policy own_rows on account_snapshots
  using (user_id = app_user_id()) with check (user_id = app_user_id());
create policy own_rows on alerts
  using (user_id = app_user_id()) with check (user_id = app_user_id());
create policy own_rows on alert_firings
  using (user_id = app_user_id()) with check (user_id = app_user_id());

-- ------------------------------------------------------------------- grants --
grant select, insert, delete on broker_connections to flint_app;
grant update (broker_user_name, status, access_token_enc, token_expires_at, last_synced_at, last_error)
  on broker_connections to flint_app;
grant select, insert, delete on broker_link_states to flint_app;
grant select, insert, delete on holdings to flint_app;              -- replaced wholesale on sync
grant select, insert, update on account_snapshots to flint_app;    -- upsert per day
grant select, insert, delete on alerts to flint_app;
grant update (threshold, status, note, triggered_at, last_checked_at) on alerts to flint_app;
grant select, insert on alert_firings to flint_app;
grant update (seen) on alert_firings to flint_app;
grant execute on function fills_paper_only() to flint_app;
