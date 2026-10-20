"""The research desk explains and never advises (product decision 2026-10-19).
No model is called: _call_json is replaced with a fake that answers each
prompt, including with injected verdict fields the desk must drop."""
import agents_desk

_FUND = {"health": {"score": 70, "grade": "B"}, "valuation": {"premium_to_own_history_pct": 10}}
_TECH = {"phase": "markup", "volume_read": "confirming", "price": 100.0}
_QUANT = {"current_price": 100.0, "composite_score": 0.3, "signal": "STRONG BUY", "conviction": 0.9,
          "position_sizing": {"recommended_fraction": 0.2}, "factors": {"rsi": 0.1},
          "levels": {"entry": 100.0, "stop_loss": 95.0, "target_1": 110.0, "target_2": 120.0, "side": "long"}}
_NEWS = {"company": "Test Co", "stance": "neutral", "weighted_score": 0.0, "articles": []}


class _Store:
    def __init__(self):
        self.saved = []

    def entries(self):
        return self.saved

    def append(self, entry):
        self.saved.append(entry)
        return entry


def _fake_call(prompt, model):
    # Every seat also tries to smuggle a call through; none may survive.
    smuggled = {"stance": "BUY", "conviction_10": 9, "size_pct": 20, "action": "BUY", "winner": "BULL"}
    if "desk's summary" in prompt:
        return {**smuggled, "summary": "Margins held.", "what_to_watch": ["Q3 results"],
                "plan_explained": "Stop is 1.0x ATR below entry."}
    if "structured debate" in prompt:
        return {**smuggled, "argument": "Revenue grew 12%.", "evidence": "12%"}
    return {**smuggled, "report": "Healthy.", "score": 0.4, "confidence": 0.8, "key_points": ["ROE 18%"]}


def _run(monkeypatch, portfolio, risk_pct):
    monkeypatch.setattr(agents_desk, "_GEMINI_KEY", "test")
    monkeypatch.setattr(agents_desk, "_call_json", _fake_call)
    monkeypatch.setattr(agents_desk.fundamentals, "get_fundamentals", lambda s: _FUND)
    monkeypatch.setattr(agents_desk.deep_analysis, "get_technical_structure", lambda s: _TECH)
    monkeypatch.setattr(agents_desk.quant_models, "get_quant_signals", lambda s: _QUANT)
    monkeypatch.setattr(agents_desk.news_intel, "get_news_intel", lambda s, max_total=20: _NEWS)
    monkeypatch.setattr(agents_desk.ml_models, "get_ml_insights", lambda s: {"fear_greed_index": 50})
    store = _Store()
    events = list(agents_desk.run_desk("TEST", portfolio=portfolio, rounds=1, store=store, risk_pct=risk_pct))
    return events, store


_BANNED = {"signal", "action", "conviction", "conviction_10", "winner", "size_pct",
           "position_sizing", "decision"}
# News carries a headline-tone "stance" in the context event; no agent may output one.
_AGENT_EVENTS = {"analyst", "debate", "summary", "rules"}


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_no_event_or_note_carries_a_call(monkeypatch):
    events, store = _run(monkeypatch, {"balance": 10000, "positions": []}, 1.0)

    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done" and "summary" in kinds and "rules" in kinds
    assert not {"judge", "trade_plan", "risk", "verdict"} & set(kinds)
    for e in events:
        assert not _BANNED & set(_keys(e)), e["type"]
        if e["type"] in _AGENT_EVENTS:
            assert "stance" not in e, e["type"]
    assert not _BANNED & set(_keys(store.saved[0]))
    assert store.saved[0]["summary"] == "Margins held."


def test_rules_are_arithmetic_on_the_users_risk(monkeypatch):
    events, _ = _run(monkeypatch, {"balance": 10000, "positions": []}, 1.0)
    rules = next(e for e in events if e["type"] == "rules")
    # 1% of 10,000 = 100 at risk; 5 per share between entry 100 and stop 95 -> 20 shares.
    assert rules["shares"] == 20 and rules["risk_amount"] == 100.0
    assert rules["pct_of_equity_after"] == 20.0 and rules["capped_by_cash"] is False


def test_rules_explain_what_is_missing(monkeypatch):
    events, _ = _run(monkeypatch, None, None)
    rules = next(e for e in events if e["type"] == "rules")
    assert rules["shares"] is None and rules["notes"]
