"""AI assistant chat and the multi-agent trading desk."""
import logging
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

import agents_desk
import ai_assistant
import llm_router

from .. import db
from ..errors import Unavailable
from ..ratelimit import limit
from ..security import Principal, require_admin
from ..stores import DbJournalStore, DbResearchRuns
from ..validators import Symbol
from . import chats
from ._common import sse

router = APIRouter(tags=["ai"])

_MAX_CONTEXT_JSON = 20_000


def _bounded_context(value: dict | None) -> dict | None:
    """Free-form portfolio snapshots are forwarded to a billed model, so their
    serialised size is capped here (audit finding 2)."""
    import json
    if value is not None and len(json.dumps(value, default=str)) > _MAX_CONTEXT_JSON:
        raise ValueError("Portfolio snapshot is too large.")
    return value


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, Field(max_length=4000)]


class ChatBody(BaseModel):
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=40)]
    portfolio: dict | None = None
    symbol: Symbol | None = None
    # The saved chat this turn continues; omitted, the turn starts a new one.
    thread_id: UUID | None = None

    _check_portfolio = field_validator("portfolio")(_bounded_context)


class DeskBody(BaseModel):
    symbol: Symbol
    portfolio: dict | None = None
    stream: bool = True

    _check_portfolio = field_validator("portfolio")(_bounded_context)


@router.post("/ai-chat")
def ai_chat(body: ChatBody, user: Principal = Depends(limit("expensive"))):
    # Checked before the model is called, so a wrong id costs no model call.
    if body.thread_id is not None:
        chats.assert_owned(user.user_id, body.thread_id)
    result = ai_assistant.chat(
        messages=[m.model_dump() for m in body.messages],
        portfolio=body.portfolio,
        symbol=body.symbol,
    )
    if "error" in result:
        raise Unavailable(result["error"])
    # History is a convenience: a store that is down never costs the reply.
    result["thread_id"] = str(body.thread_id) if body.thread_id else None
    question = body.messages[-1]
    if question.role == "user":
        try:
            result["thread_id"] = chats.save_turn(user.user_id, body.thread_id, question.content,
                                                  result["reply"], body.symbol)
        except Exception:
            logging.getLogger(__name__).exception("chat history write failed")
    return result


@router.post("/agents/run")
def agents_run(body: DeskBody, user: Principal = Depends(limit("agents"))):
    # The desk keeps notes only in this user's journal, and sizes with this
    # user's own risk-per-trade setting (None before onboarding).
    store = DbJournalStore(user.user_id)
    with db.user_tx(user.user_id) as conn:
        row = conn.execute("select risk_per_trade_pct from users").fetchone()
    risk_pct = float(row["risk_per_trade_pct"]) if row and row["risk_per_trade_pct"] is not None else None
    runs = DbResearchRuns()
    if not body.stream:
        return agents_desk.run_desk_sync(body.symbol, portfolio=body.portfolio,
                                         store=store, risk_pct=risk_pct, runs=runs)
    return sse(agents_desk.run_desk(body.symbol, portfolio=body.portfolio,
                                    store=store, risk_pct=risk_pct, runs=runs))


@router.get("/agents/routing")
def agents_routing(user: Principal = Depends(limit("standard"))):
    """Operator diagnostics: where model calls are going. Admins only (audit finding 12)."""
    require_admin(user)
    return llm_router.status()
