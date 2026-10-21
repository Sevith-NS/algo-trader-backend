-- Fills in any currency against an account in its base currency.
-- fx_rate converts one unit of the fill currency into the account currency at
-- fill time, so cash and cost basis stay exact without re-pricing history.
alter table fills
  add column fx_rate numeric(24, 10) not null default 1 check (fx_rate > 0),
  -- Client-generated id: a double-submitted order is recorded once.
  add column client_order_id uuid;

create unique index fills_client_order on fills (user_id, client_order_id)
  where client_order_id is not null;
