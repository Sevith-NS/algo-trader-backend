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
