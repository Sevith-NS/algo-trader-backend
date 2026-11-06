"""Ask Polaris streams its reply: the gateway's SSE deltas are read as they
come, the Gemini fallback is used only before the first piece, and the route
saves the whole reply (and names a new chat) once the stream is done."""
import json

import pytest

import ai_assistant
import llm_router
from conftest import auth
from flint_api.routers import chats

ASK = {"messages": [{"role": "user", "content": "How is ZZ doing?"}], "stream": True}


class _Resp:
    """A requests response that streams the given lines."""

    def __init__(self, lines, content_type="text/event-stream", status=200, text=""):
        self.lines, self.status_code, self.text = lines, status, text
        self.headers = {"content-type": content_type}

    def iter_lines(self):
        return (line.encode() for line in self.lines)

    def json(self):
        return json.loads(self.text)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def gateway(monkeypatch):
    monkeypatch.setattr(llm_router, "ENABLED", True)
    monkeypatch.setattr(llm_router, "is_available", lambda force=False: True)

    def answer(resp):
        monkeypatch.setattr(llm_router.requests, "post", lambda *a, **kw: resp)
    return answer


def test_gateway_deltas_come_out_as_they_arrive(gateway):
    gateway(_Resp([": keep-alive", 'data: {"choices":[{"delta":{"role":"assistant"}}]}',
                   'data: {"choices":[{"delta":{"content":"Margins "}}]}', "",
                   'data: {"choices":[{"delta":{"content":"held at ₹12."}}]}', "data: [DONE]"]))
    assert list(llm_router.stream_chat([])) == ["Margins ", "held at ₹12."]


def test_gateway_errors_and_whole_answers(gateway):
    gateway(_Resp(['data: {"error":{"message":"upstream quota"}}']))
    with pytest.raises(llm_router.FallbackFailed):
        list(llm_router.stream_chat([]))
    gateway(_Resp([], status=502, text="bad gateway"))
    with pytest.raises(llm_router.FallbackFailed):
        list(llm_router.stream_chat([]))
    # A gateway that ignores stream:true sends one JSON body: still an answer.
    gateway(_Resp([], content_type="application/json", text='{"choices":[{"message":{"content":" Whole. "}}]}'))
    assert list(llm_router.stream_chat([])) == ["Whole."]


class _Chunk:
    def __init__(self, text):
        self.text = text


@pytest.fixture()
def assistant(monkeypatch):
    monkeypatch.setattr(ai_assistant, "_mentioned", lambda text: [])
    monkeypatch.setattr(ai_assistant, "_build_context", lambda portfolio, symbols: "CONTEXT")
    monkeypatch.setattr(llm_router, "PRIMARY", True)
    gemini = []

    class _Session:
        def send_message_stream(self, prompt):
            gemini.append(prompt)
            return [_Chunk("From "), _Chunk(None), _Chunk("Gemini.")]
    monkeypatch.setattr(ai_assistant, "_client", object())
    monkeypatch.setattr(ai_assistant, "_gemini_session", lambda history: _Session())
    return gemini


def test_falls_back_to_gemini_only_before_the_first_piece(monkeypatch, assistant):
    def absent(turns):
        raise llm_router.RouterUnavailable("down")
        yield  # a generator, like the real one
    monkeypatch.setattr(llm_router, "stream_chat", absent)
    assert list(ai_assistant.chat_stream(ASK["messages"])) == ["From ", "Gemini."]
    assert "CONTEXT" in assistant[0]

    def cut(turns):
        yield "Half an "
        raise llm_router.FallbackFailed("dropped")
    monkeypatch.setattr(llm_router, "stream_chat", cut)
    pieces = []
    with pytest.raises(llm_router.FallbackFailed):
        for piece in ai_assistant.chat_stream(ASK["messages"]):
            pieces.append(piece)
    assert pieces == ["Half an "] and len(assistant) == 1  # no second voice mid-answer


def _events(r):
    return [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]


def test_route_streams_then_saves_and_names_the_chat(client, monkeypatch):
    monkeypatch.setattr(ai_assistant, "configured", lambda: True)
    monkeypatch.setattr(ai_assistant, "chat_stream", lambda messages, portfolio=None, symbol=None: iter(["Margins ", "held."]))
    saved, named = [], []
    monkeypatch.setattr(chats, "save_turn", lambda *a: saved.append(a) or "thread-1")
    monkeypatch.setattr(chats, "name_thread", lambda *a: named.append(a))

    r = client.post("/api/ai-chat", headers=auth(), json=ASK)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert _events(r) == [{"type": "delta", "text": "Margins "}, {"type": "delta", "text": "held."},
                          {"type": "done", "thread_id": "thread-1"}]
    assert saved[0][2:4] == ("How is ZZ doing?", "Margins held.")
    assert named == [("user_test_a", "thread-1", "How is ZZ doing?", "Margins held.")]


def test_route_ends_a_failed_stream_with_an_error_and_saves_nothing(client, monkeypatch):
    def broken(messages, portfolio=None, symbol=None):
        yield "Half"
        raise RuntimeError("model went away")
    monkeypatch.setattr(ai_assistant, "configured", lambda: True)
    monkeypatch.setattr(ai_assistant, "chat_stream", broken)
    monkeypatch.setattr(chats, "save_turn", lambda *a: pytest.fail("a broken reply is not kept"))

    events = _events(client.post("/api/ai-chat", headers=auth(), json=ASK))
    assert events == [{"type": "delta", "text": "Half"}, {"type": "error", "message": ai_assistant.UNAVAILABLE}]

    monkeypatch.setattr(ai_assistant, "configured", lambda: False)
    assert client.post("/api/ai-chat", headers=auth(), json=ASK).status_code == 503
