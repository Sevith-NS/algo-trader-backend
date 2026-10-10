"""Per-symbol research: signals, forecasts, fundamentals, news, deep analysis."""
from fastapi import APIRouter, Depends, Response

import deep_analysis
import fundamentals
import ml_models
import news_engine
import news_intel
import quant_models

from ..ratelimit import limit
from ..validators import Symbol
from ._common import cache, unwrap

router = APIRouter(tags=["symbol"])
standard = Depends(limit("standard"))
expensive = Depends(limit("expensive"))


@router.get("/quant-signals", dependencies=[standard])
def quant_signals(response: Response, symbol: Symbol):
    result = unwrap(quant_models.get_quant_signals(symbol))
    cache(response, 300)
    return result


@router.get("/ml-insights", dependencies=[standard])
def ml_insights(response: Response, symbol: Symbol):
    result = unwrap(ml_models.get_ml_insights(symbol))
    cache(response, 600)
    return result


@router.get("/advanced-signals", dependencies=[standard])
def advanced_signals(response: Response, symbol: Symbol):
    result = unwrap(ml_models.get_advanced_signals(symbol))
    cache(response, 600)
    return result


@router.get("/fundamentals", dependencies=[standard])
def fundamentals_xray(response: Response, symbol: Symbol):
    result = unwrap(fundamentals.get_fundamentals(symbol))
    cache(response, 3600)
    return result


@router.get("/news/ticker", dependencies=[standard])
def ticker_news(response: Response, symbol: Symbol):
    cache(response, 300)
    return news_engine.get_ticker_news(symbol)


@router.get("/news-intel", dependencies=[standard])
def ticker_news_intel(response: Response, symbol: Symbol):
    result = unwrap(news_intel.get_news_intel(symbol))
    cache(response, 300)
    return result


@router.get("/deep-analysis", dependencies=[expensive])
def deep_analysis_route(symbol: Symbol, ai: bool = False):
    return unwrap(deep_analysis.get_deep_analysis(symbol, include_ai=ai))
