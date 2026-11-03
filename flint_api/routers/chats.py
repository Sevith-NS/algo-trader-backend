"""Ask Polaris chat history: each user's conversations, newest first.

/ai-chat keeps every answered turn (the question and the reply) through
save_turn; these routes list, reopen and delete them. Row-level security
scopes every query to the caller, so another user's chat is simply not found.
"""
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from .. import db
from ..errors import NotFound
from ..ratelimit import limit
from ..security import Principal

router = APIRouter(prefix="/chats", tags=["ai"])

_MAX_TITLE = 80
_MAX_CONTENT = 16000  # the column's check; a reply is cut there rather than refused


def _title(question: str) -> str:
    text = " ".join(question.split())
    return text if len(text) <= _MAX_TITLE else text[:_MAX_TITLE - 1].rstrip() + "…"


def assert_owned(user_id: str, thread_id: UUID) -> None:
    with db.user_tx(user_id) as conn:
        if conn.execute("select 1 from chat_threads where id = %s", (thread_id,)).fetchone() is None:
            raise NotFound("Chat not found.")


def save_turn(user_id: str, thread_id: UUID | None, question: str, reply: str, symbol: str | None) -> str:
    """Keep one answered turn. A new chat starts a thread titled by its first question."""
    with db.user_tx(user_id) as conn:
        if thread_id is None:
            thread_id = conn.execute(
                "insert into chat_threads (user_id, title, symbol) values (%s, %s, %s) returning id",
                (user_id, _title(question), symbol)).fetchone()["id"]
        else:
            conn.execute("update chat_threads set updated_at = now() where id = %s", (thread_id,))
        for role, content in (("user", question), ("assistant", reply)):
            conn.execute("insert into chat_messages (thread_id, user_id, role, content) values (%s, %s, %s, %s)",
                         (thread_id, user_id, role, content[:_MAX_CONTENT]))
    return str(thread_id)


@router.get("")
def list_chats(user: Principal = Depends(limit("standard")),
               limit_: Annotated[int, Query(alias="limit", ge=1, le=100)] = 30):
    with db.user_tx(user.user_id) as conn:
        rows = conn.execute(
            "select t.id, t.title, t.symbol, t.updated_at, count(m.id) as messages from chat_threads t "
            "left join chat_messages m on m.thread_id = t.id "
            "group by t.id order by t.updated_at desc limit %s", (limit_,)).fetchall()
    return {"chats": [{**r, "id": str(r["id"])} for r in rows]}


@router.get("/{thread_id}")
def get_chat(thread_id: UUID, user: Principal = Depends(limit("standard"))):
    with db.user_tx(user.user_id) as conn:
        thread = conn.execute("select id, title, symbol, updated_at from chat_threads where id = %s",
                              (thread_id,)).fetchone()
        if thread is None:
            raise NotFound("Chat not found.")
        messages = conn.execute("select role, content, created_at from chat_messages "
                                "where thread_id = %s order by created_at", (thread_id,)).fetchall()
    return {**thread, "id": str(thread["id"]), "messages": messages}


@router.delete("/{thread_id}", status_code=204)
def delete_chat(thread_id: UUID, user: Principal = Depends(limit("write"))):
    with db.user_tx(user.user_id) as conn:
        if conn.execute("delete from chat_threads where id = %s", (thread_id,)).rowcount == 0:
            raise NotFound("Chat not found.")
