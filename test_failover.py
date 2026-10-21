"""End-to-end failover test: Gemini 429 -> OmniRoute, without OmniRoute installed.

Stands a stub OpenAI-compatible gateway on the port llm_router expects, forces
the Gemini path to raise a real-shaped quota error, and asserts the desk keeps
answering. Also proves the fallback stays dormant while Gemini is healthy and
that an absent gateway reproduces the original behaviour exactly.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import agents_desk
import llm_router

CALLS = {"models": 0, "chat": 0, "fenced": False}


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        CALLS["models"] += 1
        body = json.dumps({"data": [{"id": "auto"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        CALLS["chat"] += 1
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        payload = {"report": "from the fallback", "score": 0.5, "confidence": 0.9,
                   "key_points": ["routed via stub"], "flags": []}
        text = json.dumps(payload)
        if CALLS["fenced"]:
            # Not every provider behind OmniRoute honours response_format.
            text = "```json\n" + text + "\n```"
        body = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def reset():
    llm_router.clear_bench(agents_desk._ROUTE_SCOPE)
    with llm_router._lock:
        llm_router._probe["ts"] = 0.0
        llm_router._probe["ok"] = False


QUOTA_ERR = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
             "your current quota', 'details': [{'retryDelay': '47s'}]}}")

gemini_calls = {"n": 0}


class FakeClient:
    """Stands in for the genai client. Mode decides what Gemini 'does'."""
    mode = "quota"

    class models:
        @staticmethod
        def generate_content(model=None, contents=None, config=None):
            gemini_calls["n"] += 1
            if FakeClient.mode == "quota":
                raise RuntimeError(QUOTA_ERR)
            class R:
                text = json.dumps({"report": "from gemini", "score": -0.1,
                                   "confidence": 0.5, "key_points": [], "flags": []})
            return R()


def main():
    # Bind an ephemeral port, never OmniRoute's 20128. Windows lets a
    # 127.0.0.1 bind shadow a live 0.0.0.0 listener, so sharing the port made
    # these results depend on which socket won rather than on the code.
    server = HTTPServer(("127.0.0.1", 0), Stub)
    port = server.server_address[1]
    llm_router.BASE_URL = f"http://127.0.0.1:{port}/v1"
    print(f"stub gateway on :{port} — real OmniRoute, if running, is untouched")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.3)

    agents_desk._get_client = lambda: FakeClient()
    agents_desk._BACKOFF_BASE = 0.01  # keep the test fast

    # --- 1. Gemini healthy: the fallback must stay completely dormant ---
    reset()
    FakeClient.mode = "ok"
    gemini_calls["n"] = 0
    before_chat = CALLS["chat"]
    out = agents_desk._call_json("prompt", "gemini-3.6-flash")
    assert out["report"] == "from gemini", out
    assert CALLS["chat"] == before_chat, "fallback was used while Gemini was healthy"
    assert not llm_router.primary_benched(agents_desk._ROUTE_SCOPE)
    print("1. healthy Gemini -> fallback dormant                     OK")

    # --- 2. Gemini 429: reroute, and honour its own retryDelay ---
    reset()
    FakeClient.mode = "quota"
    gemini_calls["n"] = 0
    out = agents_desk._call_json("prompt", "gemini-3.6-flash")
    assert out["report"] == "from the fallback", out
    assert gemini_calls["n"] == 1, f"should not retry Gemini once rerouted, got {gemini_calls['n']}"
    remaining = llm_router.bench_remaining(agents_desk._ROUTE_SCOPE)
    assert 44 < remaining <= 47, f"retryDelay not honoured: {remaining}"
    print(f"2. 429 -> rerouted, Gemini benched {remaining:.0f}s (asked 47s)   OK")

    # --- 3. Subsequent seats skip Gemini entirely while it is benched ---
    gemini_calls["n"] = 0
    for _ in range(3):
        assert agents_desk._call_json("p", "gemini-3.6-flash")["report"] == "from the fallback"
    assert gemini_calls["n"] == 0, "benched Gemini was still being called"
    print("3. benched window -> 3 seats served, 0 Gemini calls        OK")

    # --- 4. A fenced ```json reply from a non-compliant provider still parses ---
    CALLS["fenced"] = True
    assert agents_desk._call_json("p", "gemini-3.6-flash")["report"] == "from the fallback"
    CALLS["fenced"] = False
    print("4. ```json-fenced fallback reply parses                    OK")

    # --- 5. Bench lapses: traffic returns to Gemini on its own ---
    llm_router.clear_bench(agents_desk._ROUTE_SCOPE)
    FakeClient.mode = "ok"
    gemini_calls["n"] = 0
    before_chat = CALLS["chat"]
    assert agents_desk._call_json("p", "gemini-3.6-flash")["report"] == "from gemini"
    assert CALLS["chat"] == before_chat, "still routing to fallback after bench lapsed"
    print("5. bench lapsed -> back on Gemini, fallback idle           OK")

    # --- 6. Never shorten a bench (daily exhaustion must outlive a per-minute 429) ---
    reset()
    llm_router.bench_primary("x", seconds=600)
    llm_router.bench_primary("x", seconds=5)
    assert llm_router.bench_remaining("x") > 590, llm_router.bench_remaining("x")
    llm_router.clear_bench("x")
    print("6. short 429 cannot release a long bench                   OK")

    # --- 7. Gateway absent: byte-for-byte the original behaviour ---
    server.shutdown()
    time.sleep(0.3)
    reset()
    FakeClient.mode = "quota"
    gemini_calls["n"] = 0
    try:
        agents_desk._call_json("p", "gemini-3.6-flash")
        raise AssertionError("should have raised once both paths were gone")
    except RuntimeError as exc:
        assert "RESOURCE_EXHAUSTED" in str(exc)
    assert gemini_calls["n"] == agents_desk._MAX_ATTEMPTS, (
        f"expected the original {agents_desk._MAX_ATTEMPTS}-attempt backoff, "
        f"got {gemini_calls['n']}")
    print(f"7. gateway absent -> original {agents_desk._MAX_ATTEMPTS}-attempt backoff, then raise  OK")

    # --- 8. And the seat degrades exactly as before, with the friendly message ---
    card = agents_desk._run_analyst(
        agents_desk._ANALYSTS[0],
        {"symbol": "AAPL", "company": "Apple", "fundamentals": {"health": {}},
         "technical": {}, "quant": {}, "news": {}, "ml": {}})
    assert card["score"] == 0.0 and "rate limit" in card["error"], card
    print("8. seat degrades with the friendly rate-limit message      OK")

    print("\nALL FAILOVER TESTS PASSED")


if __name__ == "__main__":
    main()
