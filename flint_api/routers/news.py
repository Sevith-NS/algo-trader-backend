"""Global news feed with sentiment."""
from fastapi import APIRouter, Depends, Response

import news_engine

from ..ratelimit import limit
from ._common import cache

router = APIRouter(tags=["news"])


@router.get("/news", dependencies=[Depends(limit("standard"))])
def global_news(response: Response):
    cache(response, 300)
    return news_engine.get_global_news()
