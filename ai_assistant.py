"""
Portfolio-aware AI trading assistant backed by Gemini.

The frontend sends the chat history plus a live snapshot of the user's paper
portfolio (and optionally the symbol currently on screen). We enrich the
prompt with live quotes and quant signals so the model grounds its answers
in real numbers instead of hallucinating prices.
"""
import os
import json
import logging

from dotenv import load_dotenv
from google import genai
from google.genai import types

import quant_models
import news_engine

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Bounds on what one chat request can send to the billed model (audit finding 2).
_MAX_HISTORY = 20
_MAX_MESSAGE_CHARS = 4000
_MAX_PORTFOLIO_CHARS = 4000
_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are Flint AI, the in-app assistant of a personal portfolio
management and stock screening platform. You help the user understand their
portfolio, markets, risk, and the platform's quant signals.

Rules:
- Ground every number you cite in the CONTEXT block provided. Never invent prices.
- Be concise and direct: short paragraphs, bullet lists for multiple points.
- When discussing a trade, always mention the risk side (stop level, position size).
- You may explain the platform's quant metrics (z-score, ATR stops, Kelly sizing,
  VaR, Sharpe) in plain language.
- Always include a one-line reminder that this is educational analysis on a paper
  trading platform, not financial advice, when giving trade-related opinions.
"""


def _build_context(portfolio: dict, symbol: str | None) -> str:
    parts = []
    if portfolio:
        # Bounded and compact: every character here is billed (audit finding 2).
        parts.append("USER PORTFOLIO SNAPSHOT:\n" + json.dumps(portfolio, default=str)[:_MAX_PORTFOLIO_CHARS])
    if symbol:
        try:
            signals = quant_models.get_quant_signals(symbol)
            if "error" not in signals:
                parts.append(f"QUANT SIGNALS FOR {symbol}:\n" + json.dumps(signals, indent=2))
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


def chat(messages: list, portfolio: dict | None = None, symbol: str | None = None):
    if not GEMINI_API_KEY:
        return {"error": "The assistant isn't configured yet."}

    try:
        context = _build_context(portfolio or {}, symbol)

        # Gemini uses 'model' for assistant turns
        history = []
        for m in messages[-_MAX_HISTORY:-1]:
            role = "model" if m.get("role") == "assistant" else "user"
            history.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=(m.get("content") or "")[:_MAX_MESSAGE_CHARS])],
            ))

        last_user = (messages[-1].get("content") or "")[:_MAX_MESSAGE_CHARS] if messages else ""
        prompt = f"CONTEXT (live data, cite from here):\n{context}\n\nUSER QUESTION:\n{last_user}"

        session = _client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            history=history,
        )
        response = session.send_message(prompt)

        return {"reply": response.text}
    except Exception:
        logging.getLogger(__name__).exception("assistant chat failed")
        return {"error": "The assistant is unavailable right now. Try again in a moment."}
