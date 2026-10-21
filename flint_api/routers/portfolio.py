"""Portfolio analytics over a list of positions.

Each symbol costs outbound calls through a process-wide admission gate, so
every list is capped at the edge (audit finding 6).
"""
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

import deep_analysis
import ml_models
import risk_analytics

from ..ratelimit import limit
from ..validators import MAX_SYMBOLS_PER_REQUEST, Symbol
from ._common import unwrap

router = APIRouter(tags=["portfolio"])


class Position(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: Symbol
    shares: Annotated[float, Field(gt=0, le=1e9)]
    averagePrice: Annotated[float, Field(ge=0, le=1e9)] = 0.0
    # Converts the holding's currency into the account currency (default: same).
    fx_rate: Annotated[float, Field(gt=0, le=1e6)] = 1.0


class PositionsBody(BaseModel):
    positions: Annotated[list[Position], Field(min_length=1, max_length=MAX_SYMBOLS_PER_REQUEST)]
    balance: Annotated[float, Field(ge=0, le=1e12)] = 0.0


class HoldingRef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: Symbol


class HoldingsBody(BaseModel):
    positions: Annotated[list[HoldingRef], Field(min_length=1, max_length=MAX_SYMBOLS_PER_REQUEST)]


class SymbolsBody(BaseModel):
    symbols: Annotated[list[Symbol], Field(min_length=2, max_length=MAX_SYMBOLS_PER_REQUEST)]


@router.post("/portfolio-analytics", dependencies=[Depends(limit("standard"))])
def portfolio_analytics(body: PositionsBody):
    positions = [p.model_dump() for p in body.positions]
    return unwrap(risk_analytics.get_portfolio_analytics(positions, body.balance))


@router.post("/portfolio-optimization", dependencies=[Depends(limit("expensive"))])
def portfolio_optimization(body: SymbolsBody):
    return unwrap(ml_models.portfolio_optimization(list(dict.fromkeys(body.symbols))))


@router.post("/holdings-intel", dependencies=[Depends(limit("expensive"))])
def holdings_intel(body: HoldingsBody):
    return deep_analysis.get_holdings_intel([p.model_dump() for p in body.positions])
