"""Event-driven backtesting: single runs and streamed parameter sweeps."""
from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict

from backtest import service as backtest_service
from backtest.errors import BacktestError

from ..errors import AppError
from ..ratelimit import limit
from ..validators import RegistryId, Symbol
from ._common import cache, sse

router = APIRouter(prefix="/backtest", tags=["backtest"])
heavy = Depends(limit("backtest"))


class BacktestBody(BaseModel):
    """Symbols are validated here; the service validates the remaining fields
    (dates, params, cost model, grid) against the strategy registry."""
    model_config = ConfigDict(extra="allow")
    symbol: Symbol
    benchmark: Symbol | None = None
    strategy: RegistryId | None = None


@router.get("/strategies", dependencies=[Depends(limit("standard"))])
def strategies(response: Response):
    cache(response, 3600)
    return {"strategies": backtest_service.list_strategies()}


@router.post("/run", dependencies=[heavy])
def run(body: BacktestBody):
    try:
        return backtest_service.run(body.model_dump(exclude_unset=True))
    except BacktestError as e:
        raise AppError(str(e))


def _stream(generator_fn, body: BacktestBody):
    # Produce the first event before streaming, so a bad grid or unknown
    # symbol is a real 400 rather than a 200 whose first frame is an error.
    try:
        stream = generator_fn(body.model_dump(exclude_unset=True))
        first = next(stream)
    except BacktestError as e:
        raise AppError(str(e))
    except StopIteration:
        raise AppError("The sweep produced no work to do.")
    return sse(_with_safe_errors(stream), first=first)


def _with_safe_errors(stream):
    try:
        yield from stream
    except BacktestError as e:
        yield {"type": "error", "message": str(e)}


@router.post("/optimize", dependencies=[heavy])
def optimize(body: BacktestBody):
    return _stream(backtest_service.optimize, body)


@router.post("/walk-forward", dependencies=[heavy])
def walk_forward(body: BacktestBody):
    return _stream(backtest_service.walk_forward, body)
