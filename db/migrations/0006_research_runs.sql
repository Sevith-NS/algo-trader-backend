-- Model output about a symbol: research-desk runs and deep-dive teardowns.
-- One row per kind, symbol, variant (the desk's round count) and IST day,
-- shared by every user because no user's data goes into it. That is also why
-- there is no row-level security here: the rows belong to no one.

create table research_runs (
  kind       text not null check (kind in ('desk', 'teardown')),
  symbol     text not null,
  variant    text not null default '',
  day        date not null,
  payload    jsonb not null,
  created_at timestamptz not null default now(),
  primary key (kind, symbol, variant, day)
);

grant select, insert on research_runs to flint_app;   -- written once, never edited
