"""AI assistant chat and the multi-agent trading desk."""
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

import agents_desk
import ai_assistant
import llm_router

from ..errors import Unavailable
from ..ratelimit import limit
from ..security import Principal, require_admin
from ..stores import DbJournalStore
from ..validators import Symbol
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

    _check_portfolio = field_validator("portfolio")(_bounded_context)


class DeskBody(BaseModel):
    symbol: Symbol
    rounds: Annotated[int, Field(ge=1, le=agents_desk.MAX_ROUNDS)] = agents_desk.DEFAULT_ROUNDS
    portfolio: dict | None = None
    stream: bool = True

    _check_portfolio = field_validator("portfolio")(_bounded_context)


@router.post("/ai-chat", dependencies=[Depends(limit("expensive"))])
def ai_chat(body: ChatBody):
    result = ai_assistant.chat(
        messages=[m.model_dump() for m in body.messages],
        portfolio=body.portfolio,
        symbol=body.symbol,
    )
    if "error" in result:
        raise Unavailable(result["error"])
    return result


@router.post("/agents/run")
def agents_run(body: DeskBody, user: Principal = Depends(limit("agents"))):
    # The desk remembers and journals only this user's decisions.
    store = DbJournalStore(user.user_id)
    if not body.stream:
        return agents_desk.run_desk_sync(body.symbol, portfolio=body.portfolio,
                                         rounds=body.rounds, store=store)
    return sse(agents_desk.run_desk(body.symbol, portfolio=body.portfolio,
                                    rounds=body.rounds, store=store))


@router.get("/agents/routing")
def agents_routing(user: Principal = Depends(limit("standard"))):
    """Operator diagnostics: where model calls are going. Admins only (audit finding 12)."""
    require_admin(user)
    return llm_router.status()
