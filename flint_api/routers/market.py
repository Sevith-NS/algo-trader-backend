"""Market-wide data: search, quotes, charts, overview, discovery, movers."""
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, Query, Response

import data_source
import levels_engine
import movers_engine
import nse_segments
import screener_engine

from ..errors import AppError, NotFound, Unavailable
from ..ratelimit import limit
from ..validators import RegistryId, Symbol
from ._common import cache

router = APIRouter(tags=["market"])
standard = Depends(limit("standard"))

_SEARCH_FIELDS = ("symbol", "shortname", "longname", "quoteType", "exchDisp", "exchange")


@router.get("/search", dependencies=[standard])
def search(response: Response, q: Annotated[str, Query(min_length=1, max_length=64)]):
    data = data_source.http_get("https://query1.finance.yahoo.com/v1/finance/search",
                                {"q": q, "quotesCount": 8, "newsCount": 0}).json()
    quotes = [{k: item.get(k) for k in _SEARCH_FIELDS if item.get(k) is not None}
              for item in (data.get("quotes") or []) if item.get("symbol")]
    cache(response, 300)
    return {"quotes": quotes}


@router.get("/quote", dependencies=[standard])
def quote(response: Response, symbol: Symbol):
    cache(response, 30)
    return data_source.get_quote(symbol)


@router.get("/chart", dependencies=[standard])
def chart(response: Response, symbol: Symbol,
          interval: Annotated[str, Query(max_length=8)] = "1d",
          range: Annotated[str, Query(max_length=8)] = "1y"):  # noqa: A002 (public param name)
    if interval not in data_source.VALID_INTERVALS:
        raise AppError(f'Unsupported interval. Use one of: {", ".join(data_source.VALID_INTERVALS)}.')
    effective = data_source.clamp_range(interval, range)
    records = data_source.get_ohlc(symbol, range_=effective, interval=interval)
    intraday = interval not in ("1d", "1wk", "1mo")
    quotes = []
    for r in records:
        ts = pd.Timestamp(r["t"])
        quotes.append({
            "time": int(ts.timestamp()) if intraday else ts.strftime("%Y-%m-%d"),
            "open": r["o"], "high": r["h"], "low": r["l"], "close": r["c"], "volume": r["v"],
        })
    cache(response, 300)
    return {"symbol": symbol, "interval": interval, "range": effective,
            "range_clamped": effective != range, "requested_range": range,
            "intraday": intraday, "count": len(quotes), "quotes": quotes}


@router.get("/trending", dependencies=[standard])
def trending(response: Response):
    cache(response, 120)
    return data_source.get_trending()


@router.get("/market-overview", dependencies=[standard])
def market_overview(response: Response):
    cache(response, 120)
    return data_source.get_market_overview()


def _scan(response: Response, result: dict, seconds: int):
    # An outage scan (nothing parsed) must not be cached past upstream recovery.
    if result.get("parsed") == 0:
        response.headers["Cache-Control"] = "no-store"
    else:
        cache(response, seconds)
    return result


@router.get("/discover", dependencies=[standard])
def discover(response: Response, universe: RegistryId = "us_large", all: bool = False):  # noqa: A002
    try:
        return _scan(response, screener_engine.get_discover(universe, include_untagged=all), 300)
    except screener_engine.UnknownUniverseError as e:
        raise AppError(str(e))


@router.get("/screens", dependencies=[standard])
def screens(response: Response):
    cache(response, 3600)
    return screener_engine.list_screens()


@router.get("/screens/run", dependencies=[standard])
def screens_run(response: Response, id: RegistryId, universe: RegistryId = "us_large"):  # noqa: A002
    try:
        return _scan(response, screener_engine.run_screen(id, universe), 300)
    except screener_engine.UnknownScreenError as e:
        raise NotFound(str(e))
    except screener_engine.UnknownUniverseError as e:
        raise AppError(str(e))


@router.get("/levels", dependencies=[standard])
def levels(response: Response, universe: RegistryId = "us_large",
           side: Literal["all", "support", "resistance"] = "all",
           limit_: Annotated[int, Query(alias="limit", ge=1, le=200)] = 40):
    try:
        return _scan(response, levels_engine.get_levels(universe, limit=limit_, side=side), 300)
    except screener_engine.UnknownUniverseError as e:
        raise AppError(str(e))


@router.get("/movers/scopes", dependencies=[standard])
def movers_scopes(response: Response):
    try:
        result = movers_engine.get_scopes()
    except nse_segments.SegmentsUnavailableError as e:
        raise Unavailable(str(e))
    cache(response, 3600)
    return result


@router.get("/movers", dependencies=[standard])
def movers(response: Response, scope: RegistryId = "allnse",
           min_move: Annotated[float, Query(ge=0, le=100)] = movers_engine.DEFAULT_MIN_MOVE,
           limit_: Annotated[int, Query(alias="limit", ge=1, le=100)] = 20):
    try:
        result = movers_engine.get_movers(scope, min_move_pct=min_move, limit=limit_)
    except nse_segments.UnknownScopeError as e:
        raise AppError(str(e))
    except nse_segments.SegmentsUnavailableError as e:
        raise Unavailable(str(e))
    # A partial board is still warming; caching it would outlive the warm.
    if result["partial"] or result["priced"] == 0:
        response.headers["Cache-Control"] = "no-store"
    else:
        cache(response, 60)
    return result
