"""
Portfolio-aware AI trading assistant backed by Gemini, or by the llm_router
gateway first (e.g. a local Ollama model) when OMNIROUTE_PRIMARY=1.

The frontend sends the chat history plus a live snapshot of the user's paper
portfolio (and optionally the symbol currently on screen). We enrich the
prompt with live quotes and quant signals so the model grounds its answers
in real numbers instead of hallucinating prices.
"""
import os
import json
import logging
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

import data_source
import llm_router
import quant_models
import news_engine

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Bounds on what one chat request can send to the billed model (audit finding 2).
_MAX_HISTORY = 20
_MAX_MESSAGE_CHARS = 4000
_MAX_PORTFOLIO_CHARS = 4000
# google-genai has no default timeout: a hung call would hold the function until Vercel kills it.
_client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=60_000)) if GEMINI_API_KEY else None

MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are the Polaris assistant, inside a read-only portfolio tracker and
paper-trading tool. You help the user understand their portfolio, markets, risk,
and how the platform's quant readings and plan levels are computed.

Rules:
- Ground every number you cite in the CONTEXT block provided. Never invent prices.
  CONTEXT holds live readings and headlines for the companies the user names. If a
  named company has no block, say its data could not be fetched; do not say you have
  no access to market data.
- Be concise and direct: short paragraphs, bullet lists for multiple points.
- You explain; you never advise. Do not tell the user to buy, sell, hold, avoid,
  add, trim or exit anything, and do not give price targets, ratings, convictions
  or position sizes of your own. If asked "should I buy X?", explain what the data
  shows on both sides and that the decision is theirs.
- You may explain the platform's quant metrics (z-score, ATR stops, R multiples,
  VaR, Sharpe) and how a plan's entry, stop and size were computed, in plain language.
- Polaris is not a SEBI-registered investment adviser or research analyst. When a
  question is about a trade, add one line saying this is information, not advice.
"""


_MAX_MENTIONED = 3
_TICKER = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=&^]{0,23}$")  # validators.SYMBOL_PATTERN, upper-case


def _mentioned(text: str) -> list:
    """Yahoo tickers for the companies a question names ("google and meta" ->
    GOOGL, META), so their data is fetched too. The model only names the
    companies (small models guess tickers badly); Yahoo's search finds the
    symbol. Best effort: [] on any failure."""
    prompt = ('List the listed companies named in this message, as the user wrote them '
              '(a ticker they typed counts). Respond as JSON: {"companies": ["..."]}, at most '
              f'{_MAX_MENTIONED}, [] if none.\n\nMESSAGE:\n{text}')
    try:
        if llm_router.PRIMARY:
            raw = llm_router.call_json(prompt)
        elif _client:
            raw = json.loads(_client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")).text)
        else:
            return []
        found = []
        for name in [str(c).strip()[:60] for c in raw.get("companies", [])][:_MAX_MENTIONED]:
            try:
                hits = data_source.http_get("https://query1.finance.yahoo.com/v1/finance/search",
                                            {"q": name, "quotesCount": 5, "newsCount": 0}).json()["quotes"]
            except Exception:
                continue
            top = next((h["symbol"].upper() for h in hits if h.get("quoteType") == "EQUITY"), None)
            if top and _TICKER.match(top):
                found.append(top)
        return found
    except Exception:
        return []


def title_for(question: str, reply: str) -> str | None:
    """Three to six words naming a chat, the way chat apps title a conversation.
    Model output is untrusted: whitespace collapsed, quotes and end punctuation
    stripped, cut to 80 characters. None when no model answers."""
    prompt = ('Name this conversation in 3 to 6 words, like a chat app\'s sidebar title. Plain words, '
              'sentence case, no quotes, no ending punctuation; keep any ticker as written. '
              'Respond as JSON: {"title": "..."}\n\n'
              f'QUESTION:\n{question[:600]}\n\nANSWER:\n{reply[:600]}')
    try:
        if llm_router.PRIMARY:
            raw = llm_router.call_json(prompt)
        elif _client:
            raw = json.loads(_client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")).text)
        else:
            return None
        title = " ".join(str(raw.get("title") or "").split()).strip(" .!?:;\"'")
        return title[:80] or None
    except Exception:
        return None


def _build_context(portfolio: dict, symbols: list) -> str:
    parts = []
    if portfolio:
        # Bounded and compact: every character here is billed (audit finding 2).
        parts.append("USER PORTFOLIO SNAPSHOT:\n" + json.dumps(portfolio, default=str)[:_MAX_PORTFOLIO_CHARS])
    for symbol in symbols:
        try:
            signals = quant_models.get_quant_signals(symbol)
            if "error" not in signals:
                # Readings and levels only: no buy/sell label, conviction or engine size
                # reaches the model, so it has none to repeat.
                readings = {k: v for k, v in signals.items()
                            if k not in ("signal", "action", "conviction", "position_sizing")}
                parts.append(f"QUANT READINGS FOR {symbol}:\n" + json.dumps(readings, indent=2))
        except Exception:
            pass
        try:
            news = news_engine.get_ticker_news(symbol, max_articles=6)
            headlines = [
                {"title": a["title"], "sentiment": a["sentiment"]}
                for a in news.get("articles", [])
            ]
            parts.append(
                f"RECENT NEWS SENTIMENT FOR {symbol} (mood {news['mood_index']}/100):\n"
                + json.dumps(headlines, indent=2)
            )
        except Exception:
            pass
    return "\n\n".join(parts) if parts else "No live context available."


UNAVAILABLE = "The assistant is unavailable right now. Try again in a moment."


def configured() -> bool:
    return bool(GEMINI_API_KEY or llm_router.PRIMARY)


def _turns(messages: list, portfolio: dict | None, symbol: str | None):
    """One question with its live context: the prompt, Gemini's history and the gateway's turns."""
    last_text = (messages[-1].get("content") or "")[:_MAX_MESSAGE_CHARS] if messages else ""
    symbols = list(dict.fromkeys([symbol, *_mentioned(last_text)] if symbol else _mentioned(last_text)))
    context = _build_context(portfolio or {}, symbols)

    # Gemini uses 'model' for assistant turns
    history = []
    for m in messages[-_MAX_HISTORY:-1]:
        role = "model" if m.get("role") == "assistant" else "user"
        history.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=(m.get("content") or "")[:_MAX_MESSAGE_CHARS])],
        ))

    prompt = f"CONTEXT (live data, cite from here):\n{context}\n\nUSER QUESTION:\n{last_text}"
    turns = [{"role": "system", "content": SYSTEM_PROMPT}]
    turns += [{"role": "assistant" if m.get("role") == "assistant" else "user",
               "content": (m.get("content") or "")[:_MAX_MESSAGE_CHARS]}
              for m in messages[-_MAX_HISTORY:-1]]
    turns.append({"role": "user", "content": prompt})
    return prompt, history, turns


def _gemini_session(history: list):
    return _client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        history=history,
    )


def chat(messages: list, portfolio: dict | None = None, symbol: str | None = None):
    if not configured():
        return {"error": "The assistant isn't configured yet."}

    try:
        prompt, history, turns = _turns(messages, portfolio, symbol)
        if llm_router.PRIMARY:
            try:
                return {"reply": llm_router.call_chat(turns)}
            except (llm_router.RouterUnavailable, llm_router.FallbackFailed):
                if _client is None:
                    raise  # no Gemini key to fall back to

        return {"reply": _gemini_session(history).send_message(prompt).text}
    except Exception:
        logging.getLogger(__name__).exception("assistant chat failed")
        return {"error": UNAVAILABLE}


def chat_stream(messages: list, portfolio: dict | None = None, symbol: str | None = None):
    """chat(), yielded piece by piece as the model writes it. The gateway falls
    back to Gemini only before its first piece; after that an error ends the
    answer where it stopped. Raises on failure: the caller words the error."""
    prompt, history, turns = _turns(messages, portfolio, symbol)
    if llm_router.PRIMARY:
        sent = False
        try:
            for piece in llm_router.stream_chat(turns):
                sent = True
                yield piece
            return
        except (llm_router.RouterUnavailable, llm_router.FallbackFailed):
            if sent or _client is None:
                raise
    for chunk in _gemini_session(history).send_message_stream(prompt):
        if chunk.text:
            yield chunk.text
