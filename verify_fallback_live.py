"""A whole desk run with Gemini hard-failed — every seat served by OmniRoute.

Needs a live `omniroute serve` on :20128 and takes several minutes, so it is
named verify_* rather than test_* — the test_*.py files in this repo are all
hermetic and safe to collect.
"""
import time
import agents_desk, llm_router

QUOTA = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'quota', "
         "'details': [{'retryDelay': '600s'}]}}")

class FakeClient:
    class models:
        @staticmethod
        def generate_content(**kw):
            raise RuntimeError(QUOTA)

agents_desk._get_client = lambda: FakeClient()
llm_router.clear_bench(agents_desk._ROUTE_SCOPE)

t0 = time.time()
seats, errors, notices = 0, [], []
for ev in agents_desk.run_desk("MSFT", portfolio={"balance": 30000, "positions": []}, rounds=1):
    k, el = ev.get("type"), f"{time.time()-t0:5.1f}s"
    if ev.get("error"):
        errors.append((k, ev.get("label") or ev.get("seat"), ev["error"][:70]))
    if k == "notice":
        notices.append(ev["message"]); print(f"{el} NOTICE  {ev['message']}")
    elif k == "analyst":
        seats += 1; print(f"{el} ANALYST {ev['label']:26s} {ev['score']:+.2f}")
    elif k == "debate":
        seats += 1; print(f"{el} DEBATE  R{ev['round']} {ev['label']:17s} {(ev['argument'] or '')[:60]}")
    elif k == "judge":
        seats += 1; print(f"{el} JUDGE   {ev['winner']}/{ev['stance']} conv={ev['conviction_10']}")
    elif k == "trade_plan":
        seats += 1; print(f"{el} TRADER  {ev['action']} {ev['horizon_days']}d")
    elif k == "risk":
        seats += 1; print(f"{el} RISK    {ev['seat']:13s} {ev['stance']:10s} {ev['size_pct']}%")
    elif k == "verdict":
        seats += 1; print(f"{el} PM      {ev['decision']} {ev['action']} {ev['size_pct']}% shares={ev['sizing']['shares']}")
    elif k == "done":
        print(f"{el} DONE    {ev['elapsed_ms']}ms routing={ev['routing']}")
    elif k == "error":
        print(f"{el} ERROR   {ev['message']}")

print(f"\nseats that filed: {seats}/11   seat errors: {errors or 'none'}")
print(f"failover notice emitted: {bool(notices)}")
print("counters:", llm_router.counters())
