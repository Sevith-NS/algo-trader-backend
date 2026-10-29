"""Polaris shows readings, never calls (product decision 2026-10-19, SEBI RA
rules): no buy/sell labels, conviction or engine-suggested size leaves the
API, and the AI narrative can't smuggle a verdict through."""
import deep_analysis
from conftest import auth
from flint_api.routers import symbol as symbol_router


def test_quant_signals_drop_calls_and_sizing(client, monkeypatch):
    monkeypatch.setattr(symbol_router.quant_models, "get_quant_signals", lambda s: {
        "symbol": s, "signal": "STRONG BUY", "action": "buy", "conviction": 0.9, "composite_score": 0.5,
        "position_sizing": {"recommended_fraction": 0.2}, "levels": {"entry": 1.0}, "factors": {"rsi": 0.1}})

    body = client.get("/api/quant-signals", params={"symbol": "ZZTEST"}, headers=auth()).json()

    assert not {"signal", "action", "conviction", "position_sizing"} & set(body)
    assert body["levels"] == {"entry": 1.0} and body["factors"] == {"rsi": 0.1}


def test_readings_carry_no_call():
    readings = deep_analysis._build_readings({"error": "x"}, {"error": "x"}, {"error": "x"}, {"weighted_score": 0.2})
    assert set(readings) == {"pillars"}
    assert readings["pillars"]["fundamentals"] is None and readings["pillars"]["news"] == 0.5


def test_narrative_keeps_only_known_sections():
    raw = {"one_liner": "Margins are flat.", "final_verdict": {"call": "BUY"}, "rating": "Strong buy",
           "bull_case": ["not", "a", "string"]}
    assert deep_analysis._sanitize_narrative(raw) == {"one_liner": "Margins are flat."}
    assert deep_analysis._sanitize_narrative(["BUY"]) is None


class _Teardowns:
    def __init__(self, row=None):
        self.row, self.writes = row, []

    def latest(self, kind, symbol, variant=""):
        return self.row

    def put(self, kind, symbol, variant, payload):
        self.writes.append(payload)
        self.row = {"payload": payload, "day": "2026-10-28", "today": True}


def _stub_engines(monkeypatch, notes):
    """No data and no model: `notes` is what the model would write, call by call."""
    monkeypatch.setattr(deep_analysis.fundamentals, "get_fundamentals", lambda s: {"error": "x"})
    monkeypatch.setattr(deep_analysis, "get_technical_structure", lambda s: {"error": "x"})
    monkeypatch.setattr(deep_analysis.quant_models, "get_quant_signals", lambda s: {"error": "x"})
    monkeypatch.setattr(deep_analysis.news_intel, "get_news_intel", lambda s, max_total=20: {
        "stance": "neutral", "rating": 50, "confidence": 0.5, "weighted_score": 0.0, "article_count": 0,
        "distinct_outlets": 0, "breakdown": {}, "articles": []})
    monkeypatch.setattr(deep_analysis, "_company_profile", lambda s: None)
    monkeypatch.setattr(deep_analysis, "_narrative", lambda *a: notes.pop(0) if notes else None)


def test_the_teardown_is_written_once_a_day_and_shown_on_open(monkeypatch):
    _stub_engines(monkeypatch, [{"one_liner": "Margins held."}])
    runs = _Teardowns()

    assert deep_analysis.get_deep_analysis("ZZ", runs=runs)["narrative"] is None
    first = deep_analysis.get_deep_analysis("ZZ", include_ai=True, runs=runs)
    again = deep_analysis.get_deep_analysis("ZZ", include_ai=True, runs=runs)  # a second write would get None
    opened = deep_analysis.get_deep_analysis("ZZ", runs=runs)

    assert first["narrative"] == again["narrative"] == opened["narrative"] == {"one_liner": "Margins held."}
    assert len(runs.writes) == 1 and opened["narrative_at"] == "2026-10-28"


def test_ask_polaris_runs_on_the_gateway_with_its_rules_and_live_context(monkeypatch):
    import ai_assistant
    sent = []
    monkeypatch.setattr(ai_assistant.llm_router, "PRIMARY", True)
    monkeypatch.setattr(ai_assistant.llm_router, "call_chat", lambda turns: sent.append(turns) or "Margins held.")
    monkeypatch.setattr(ai_assistant, "_mentioned", lambda text: ["ZZ", "YY"])
    contexts = []
    monkeypatch.setattr(ai_assistant, "_build_context",
                        lambda portfolio, symbols: contexts.append(symbols) or "QUANT READINGS FOR ZZ: {}")

    reply = ai_assistant.chat([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
                               {"role": "user", "content": "How is ZZ doing?"}], symbol="ZZ")

    turns = sent[0]
    assert contexts == [["ZZ", "YY"]]  # the open symbol plus the one named in the question, once each
    assert reply == {"reply": "Margins held."}
    assert turns[0] == {"role": "system", "content": ai_assistant.SYSTEM_PROMPT}
    assert [t["role"] for t in turns[1:]] == ["user", "assistant", "user"]
    assert "QUANT READINGS FOR ZZ" in turns[-1]["content"] and "How is ZZ doing?" in turns[-1]["content"]

    def down(turns):
        raise ai_assistant.llm_router.RouterUnavailable("down")
    monkeypatch.setattr(ai_assistant.llm_router, "call_chat", down)
    monkeypatch.setattr(ai_assistant, "_client", None)
    assert "error" in ai_assistant.chat([{"role": "user", "content": "Hi"}])


def test_a_failed_rewrite_never_passes_off_the_old_teardown(monkeypatch):
    _stub_engines(monkeypatch, [])
    runs = _Teardowns({"payload": {"one_liner": "Old."}, "day": "2026-10-01", "today": False})

    assert deep_analysis.get_deep_analysis("ZZ", include_ai=True, runs=runs)["narrative"] is None
    opened = deep_analysis.get_deep_analysis("ZZ", runs=runs)
    assert opened["narrative"] == {"one_liner": "Old."} and opened["narrative_at"] == "2026-10-01"
