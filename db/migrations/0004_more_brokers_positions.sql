-- Polaris v3.1: more read-only brokers (FYERS, Angel One, Dhan) and the open
-- positions of each real account, synced with its holdings. Positions feed the
-- broker watchlist; nothing here can place an order.

alter table broker_connections drop constraint broker_connections_broker_check;
alter table broker_connections add constraint broker_connections_broker_check
  check (broker in ('zerodha', 'fyers', 'angel', 'dhan'));
alter table broker_link_states drop constraint broker_link_states_broker_check;
alter table broker_link_states add constraint broker_link_states_broker_check
  check (broker in ('zerodha', 'fyers', 'angel', 'dhan'));

-- Today's net positions of a real account, replaced on every sync. Quantity is
-- signed (short intraday positions are negative) and 0 for one closed today.
create table broker_positions (
  account_id     uuid not null,
  user_id        text not null,
  symbol         text not null check (symbol ~ '^[A-Z0-9^][A-Z0-9.=&^-]{0,23}$'),
  broker_symbol  text not null check (length(broker_symbol) between 1 and 40),
  exchange       text not null check (exchange in ('NSE', 'BSE')),
  product        text not null check (length(product) between 1 and 20),
  quantity       numeric(24, 8) not null,
  avg_price      numeric(24, 8) not null check (avg_price >= 0),
  last_price     numeric(24, 8) check (last_price is null or last_price >= 0),
  close_price    numeric(24, 8) check (close_price is null or close_price >= 0),
  currency       char(3) not null check (currency ~ '^[A-Z]{3}$'),
  synced_at      timestamptz not null,
  primary key (account_id, symbol, product),
  foreign key (account_id, user_id) references accounts (id, user_id) on delete cascade
);

alter table broker_positions enable row level security;
alter table broker_positions force row level security;
create policy own_rows on broker_positions
  using (user_id = app_user_id()) with check (user_id = app_user_id());

grant select, insert, delete on broker_positions to flint_app;   -- replaced wholesale on sync
