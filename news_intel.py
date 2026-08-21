"""
Multi-source news intelligence for a single ticker.

Where news_engine.py answers "what's the market mood?", this module answers
"what is being written about THIS stock, everywhere, and what does it net to?"

  Sources (region-aware):
    - Google News RSS, broad query        (IN edition for .NS/.BO tickers)
    - Google News RSS, earnings-focused   ("results OR earnings OR profit")
    - Bing News RSS
    - Yahoo Finance RSS                   (US/global tickers only — empty for NSE)

  Scoring:
    - VADER with a finance-tuned lexicon overlay (upgrade/downgrade, beats/
      misses, fraud/probe, pledge, multibagger...) — stock headlines are not
      movie reviews; base VADER misses half the signal.
    - Recency-weighted aggregate (3-day decay half-life): yesterday's downgrade
      matters more than a puff piece from three weeks ago.
    - Stance = bullish / bearish / neutral + a 0-100 rating and a confidence
      level derived from article count, source diversity and score agreement.
"""
import math
import re
import time
import urllib.parse
from email.utils import parsedate_to_datetime

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import data_source

_analyzer = SentimentIntensityAnalyzer()
# Finance overlay: VADER's default lexicon is tuned for social text, not tape.
_analyzer.lexicon.update({
    "upgrade": 2.3, "upgraded": 2.3, "downgrade": -2.6, "downgraded": -2.6,
    "beats": 2.0, "beat": 1.6, "misses": -2.2, "missed": -1.8,
    "surge": 2.4, "surges": 2.4, "soars": 2.8, "soar": 2.8, "jumps": 2.0,
    "plunge": -2.8, "plunges": -2.8, "tumbles": -2.4, "slumps": -2.2,
    "sinks": -2.2, "crashes": -3.2, "crash": -3.0, "selloff": -2.4,
    "rally": 2.2, "rallies": 2.2, "rebound": 1.8, "breakout": 2.0,
    "bullish": 2.4, "bearish": -2.4, "outperform": 2.0, "underperform": -2.0,
    "overweight": 1.8, "underweight": -1.8, "buyback": 1.6, "bonus": 1.4,
    "dividend": 0.9, "record": 1.5, "multibagger": 2.4, "rerating": 1.8,
    "fraud": -3.4, "probe": -2.2, "investigation": -2.0, "lawsuit": -1.9,
    "penalty": -1.8, "default": -3.0, "bankruptcy": -3.4, "insolvency": -3.2,
    "pledge": -1.4, "pledged": -1.4, "downtrend": -1.8, "uptrend": 1.8,
    "downgrades": -2.6, "upgrades": 2.3, "slashes": -2.2, "raises": 1.6,
    "cuts": -1.6, "weak": -1.4, "strong": 1.4, "robust": 1.6, "muted": -1.2,
})

_cache: dict = {}
CACHE_TTL = 600  # 10 min per symbol


def _is_indian(symbol: str) -> bool:
    return symbol.upper().endswith((".NS", ".BO"))


def _google_rss(query: str, indian: bool) -> str:
    region = "hl=en-IN&gl=IN&ceid=IN:en" if indian else "hl=en-US&gl=US&ceid=US:en"
    return f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&{region}"


def _fetch_feed(url: str, source_tag: str, max_items: int = 15) -> list:
    try:
        resp = data_source.http_get(url, timeout=12)
        feed = feedparser.parse(resp.content)
    except Exception:
        return []
    items = []
    for entry in feed.entries[:max_items]:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        published = None
        if entry.get("published"):
            try:
                published = parsedate_to_datetime(entry.published)
            except Exception:
                pass
        outlet = None
        if entry.get("source") and entry.source.get("title"):
            outlet = entry.source.title
        items.append({
            "title": title,
            "link": entry.get("link"),
            "source": outlet or source_tag,
            "feed": source_tag,
            "published_dt": published,
        })
    return items


def _norm_title(title: str) -> str:
    """Dedup key: lowercase, outlet suffix stripped, non-alnum collapsed."""
    t = title.rsplit(" - ", 1)[0].lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _stance_label(score: float) -> str:
    if score >= 0.15:
        return "bullish"
    if score <= -0.15:
        return "bearish"
    return "neutral"


def get_news_intel(symbol: str, max_total: int = 30) -> dict:
    now = time.time()
    cached = _cache.get(symbol)
    if cached and now - cached["ts"] < CACHE_TTL:
        return cached["data"]

    indian = _is_indian(symbol)
    base = symbol.split(".")[0]

    # Resolve a printable company name (quote is cached in data_source)
    quote = data_source.get_quote(symbol)
    name = quote.get("longName") or quote.get("shortName") or base
    # "Reliance Industries Limited" → "Reliance Industries" for better queries
    query_name = re.sub(r"\b(limited|ltd\.?|inc\.?|corp\.?|corporation|plc)\b\.?$",
                        "", name, flags=re.I).strip() or base

    feeds = [
        (_google_rss(f'"{query_name}" stock', indian), "Google News"),
        (_google_rss(f'"{query_name}" results OR earnings OR profit', indian), "Google News (earnings)"),
        (f"https://www.bing.com/news/search?q={urllib.parse.quote(query_name + ' stock')}&format=RSS", "Bing News"),
    ]
    if not indian:
        feeds.append((
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(symbol)}&region=US&lang=en-US",
            "Yahoo Finance"))

    seen, articles = set(), []
    sources_scanned = [tag for _, tag in feeds]
    for url, tag in feeds:
        for item in _fetch_feed(url, tag):
            key = _norm_title(item["title"])
            if not key or key in seen:
                continue
            seen.add(key)
            articles.append(item)

    # Score + recency weight
    now_ts = time.time()
    scored = []
    for a in articles:
        score = _analyzer.polarity_scores(a["title"])["compound"]
        age_days = 30.0
        published_iso = None
        if a["published_dt"] is not None:
            published_iso = a["published_dt"].isoformat()
            age_days = max(0.0, (now_ts - a["published_dt"].timestamp()) / 86400)
        weight = math.exp(-age_days / 3.0)  # 3-day decay
        scored.append({
            "title": a["title"],
            "link": a["link"],
            "source": a["source"],
            "feed": a["feed"],
            "published": published_iso,
            "age_days": round(age_days, 1),
            "sentiment_score": round(score, 3),
            "sentiment": _stance_label(score),
            "weight": round(weight, 3),
        })

    scored.sort(key=lambda a: a["published"] or "", reverse=True)
    scored = scored[:max_total]

    total_w = sum(a["weight"] for a in scored)
    weighted = (sum(a["sentiment_score"] * a["weight"] for a in scored) / total_w
                if total_w > 0 else 0.0)

    distinct_outlets = {a["source"] for a in scored}
    n = len(scored)
    if n >= 2:
        mean = sum(a["sentiment_score"] for a in scored) / n
        variance = sum((a["sentiment_score"] - mean) ** 2 for a in scored) / n
        agreement = max(0.0, 1.0 - min(1.0, math.sqrt(variance) / 0.5))
    else:
        agreement = 0.0
    confidence_score = (min(1.0, n / 20) * 0.5
                        + min(1.0, len(distinct_outlets) / 4) * 0.3
                        + agreement * 0.2)
    confidence = ("high" if confidence_score >= 0.65
                  else "medium" if confidence_score >= 0.4 else "low")

    bull = sum(1 for a in scored if a["sentiment"] == "bullish")
    bear = sum(1 for a in scored if a["sentiment"] == "bearish")

    data = {
        "symbol": symbol,
        "company": name,
        "articles": scored,
        "article_count": n,
        "sources_scanned": sources_scanned,
        "distinct_outlets": len(distinct_outlets),
        "weighted_score": round(weighted, 3),
        "stance": _stance_label(weighted),
        "rating": int((weighted + 1) * 50),  # 0-100
        "confidence": confidence,
        "breakdown": {"bullish": bull, "bearish": bear, "neutral": n - bull - bear},
        "generated_at": now_ts,
    }
    _cache[symbol] = {"ts": now, "data": data}
    return data
