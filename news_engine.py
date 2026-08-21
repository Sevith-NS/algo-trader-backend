"""
Global market news aggregator with per-headline sentiment scoring.

Pulls Google News RSS feeds across market categories, scores each headline
with VADER, and aggregates a market mood index per category and overall.
Results are cached in-process for 5 minutes to keep the UI snappy.
"""
import time
import urllib.parse
from email.utils import parsedate_to_datetime

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

CATEGORIES = {
    "markets": "stock market today",
    "economy": "global economy inflation fed",
    "crypto": "cryptocurrency bitcoin market",
    "commodities": "oil gold commodities prices",
    "forex": "forex currency dollar",
    "asia": "asia markets nikkei china economy",
    "europe": "europe markets ECB economy",
    "tech": "technology stocks earnings",
}

_cache = {"data": None, "ts": 0}
CACHE_TTL = 300  # seconds


def _sentiment_label(score: float) -> str:
    if score >= 0.35:
        return "bullish"
    if score >= 0.05:
        return "slightly-bullish"
    if score <= -0.35:
        return "bearish"
    if score <= -0.05:
        return "slightly-bearish"
    return "neutral"


def _fetch_category(key: str, query: str, max_articles: int = 10):
    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_articles]:
        title = entry.get("title", "")
        score = _analyzer.polarity_scores(title)["compound"]
        published = None
        if entry.get("published"):
            try:
                published = parsedate_to_datetime(entry.published).isoformat()
            except Exception:
                pass
        source = None
        if entry.get("source") and entry.source.get("title"):
            source = entry.source.title
        articles.append({
            "title": title,
            "link": entry.get("link"),
            "source": source,
            "published": published,
            "sentiment_score": round(score, 3),
            "sentiment": _sentiment_label(score),
            "category": key,
        })
    return articles


def get_global_news():
    now = time.time()
    if _cache["data"] and now - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    categories = {}
    all_articles = []
    for key, query in CATEGORIES.items():
        try:
            articles = _fetch_category(key, query)
        except Exception:
            articles = []
        scores = [a["sentiment_score"] for a in articles]
        avg = sum(scores) / len(scores) if scores else 0.0
        categories[key] = {
            "articles": articles,
            "avg_sentiment": round(avg, 3),
            # Map [-1, 1] onto a 0-100 mood index
            "mood_index": int((avg + 1) * 50),
        }
        all_articles.extend(articles)

    overall_scores = [a["sentiment_score"] for a in all_articles]
    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    all_articles.sort(key=lambda a: a["published"] or "", reverse=True)

    data = {
        "overall_mood_index": int((overall + 1) * 50),
        "overall_sentiment": _sentiment_label(overall),
        "categories": categories,
        "latest": all_articles[:40],
        "generated_at": now,
    }
    _cache["data"] = data
    _cache["ts"] = now
    return data


def get_ticker_news(symbol: str, max_articles: int = 12):
    """Headline feed + sentiment for a single ticker (used by screener + AI bot)."""
    articles = _fetch_category("ticker", f"{symbol} stock", max_articles)
    scores = [a["sentiment_score"] for a in articles]
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "symbol": symbol,
        "articles": articles,
        "avg_sentiment": round(avg, 3),
        "mood_index": int((avg + 1) * 50),
        "sentiment": _sentiment_label(avg),
    }
