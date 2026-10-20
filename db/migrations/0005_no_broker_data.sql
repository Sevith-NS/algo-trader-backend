-- Nothing about a broker is stored any more. Connections are sealed cookies in
-- the user's browser (flint_api/broker_jar.py) and holdings and positions are
-- read live from the broker, so every broker table goes, with the real
-- accounts and their daily snapshots. Accounts are paper only again.

delete from accounts where kind = 'real';

drop table broker_positions;
drop table holdings;
drop table account_snapshots;
drop table broker_link_states;
alter table accounts drop constraint accounts_broker_connection;
drop table broker_connections;

alter table accounts drop constraint accounts_kind_shape;
alter table accounts drop column broker_connection_id;   -- drops accounts_one_per_connection
alter table accounts add constraint accounts_kind_shape check (
  kind = 'paper' and starting_cash is not null and starting_cash > 0 and starting_cash <= 1e12);
