-- Ask Polaris conversations, kept per user so a chat can be reopened later.
-- Each answered turn stores the question and the reply. Row-level security
-- as on every other user table; the composite key keeps a message on a thread
-- of the same user.

create table chat_threads (
  id          uuid primary key default gen_random_uuid(),
  user_id     text not null references users (id) on delete cascade,
  title       text not null check (length(title) between 1 and 120),
  symbol      text check (symbol is null or symbol ~ '^[A-Z0-9^][A-Z0-9.=&^-]{0,23}$'),
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  unique (id, user_id)
);
create index chat_threads_recent on chat_threads (user_id, updated_at desc);

create table chat_messages (
  id          uuid primary key default gen_random_uuid(),
  thread_id   uuid not null,
  user_id     text not null,
  role        text not null check (role in ('user', 'assistant')),
  content     text not null check (length(content) between 1 and 16000),
  -- clock_timestamp, not now(): a question and its reply share a transaction.
  created_at  timestamptz not null default clock_timestamp(),
  foreign key (thread_id, user_id) references chat_threads (id, user_id) on delete cascade
);
create index chat_messages_thread on chat_messages (thread_id, created_at);

alter table chat_threads  enable row level security;
alter table chat_messages enable row level security;
alter table chat_threads  force row level security;
alter table chat_messages force row level security;

create policy own_rows on chat_threads
  using (user_id = app_user_id()) with check (user_id = app_user_id());
create policy own_rows on chat_messages
  using (user_id = app_user_id()) with check (user_id = app_user_id());

grant select, insert, delete on chat_threads to flint_app;
grant update (title, updated_at) on chat_threads to flint_app;
grant select, insert on chat_messages to flint_app;
