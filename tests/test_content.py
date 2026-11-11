from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent_control_panel import MAX_FULL_CHARS, Request, Response, extract_text
from agent_control_panel._content import cut_utf16
from agent_control_panel._reporter import MAX_BATCH_BYTES, chunk_by_size

from .conftest import ClientFactory, FakeTransport


def test_answer_text_from_sdk_objects() -> None:
    openai_chat = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi from gpt"))])
    assert extract_text(openai_chat) == "hi from gpt"
    parts = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=[{"type": "text", "text": "a"}, "b"]))]
    )
    assert extract_text(parts) == "ab"
    assert extract_text(SimpleNamespace(output_text="responses api")) == "responses api"
    anthropic = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", text="no"),
            SimpleNamespace(type="text", text="claude says"),
        ]
    )
    assert extract_text(anthropic) == "claude says"
    gemini = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="gem")]))]
    )
    assert extract_text(gemini) == "gem"
    wrapped = SimpleNamespace(
        response={"candidates": [{"content": {"parts": [{"text": "x"}, {"text": "y"}]}}]}
    )
    assert extract_text(wrapped) == "xy"
    assert extract_text({"text": "vercel"}) == "vercel"
    assert extract_text("plain") == "plain"


def test_answer_text_never_calls_anything() -> None:
    called: list[str] = []

    class Result:
        def text(self) -> str:
            called.append("text")
            return "from a method"

        @property
        def output_text(self) -> str:
            raise RuntimeError("lazy property failed")

    assert extract_text(Result()) is None
    assert called == []


@pytest.mark.parametrize(
    "junk", [None, "", 42, [], {}, {"choices": []}, {"content": "str"}, {"candidates": [None]}]
)
def test_answer_text_for_unknown_shapes(junk: Any) -> None:
    assert extract_text(junk) is None


def test_cut_counts_utf16_units_and_never_splits_a_character() -> None:
    assert cut_utf16("abc", 2) == ("ab", True)
    assert cut_utf16("😀😀", 3) == ("😀", True)
    assert cut_utf16("😀😀", 4) == ("😀😀", False)
    assert cut_utf16("a" * MAX_FULL_CHARS, MAX_FULL_CHARS) == ("a" * MAX_FULL_CHARS, False)


def test_a_wrapped_run_reports_the_answer_and_cuts_the_summary(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport)
    long_answer = "z" * 3_000

    async def call() -> dict[str, Any]:
        return {"choices": [{"message": {"content": long_answer}}]}

    asyncio.run(acp.wrap_agent_run_async(call, user_input="q", full_input=[{"role": "user", "content": "q"}]))
    acp.stop()
    [run] = transport.bodies("/api/report")[0]["runs"]
    assert run["output_summary"] == "z" * 2_000
    assert run["full_output"] == long_answer
    assert run["user_input"] == "q"
    assert json.loads(run["full_input"]) == [{"role": "user", "content": "q"}]


def test_capture_off_keeps_content_out_of_every_request(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport, capture_content=False)
    acp.wrap_agent_run(
        lambda: {"output_text": "secret answer"}, user_input="secret question", full_input="secret"
    )
    acp.report_run(
        "success", user_input="secret question", full_output="secret answer", input_summary="secret"
    )
    acp.stop()
    raw = b"".join(r.body or b"" for r in transport.requests)
    assert b"secret" not in raw


def test_large_flushes_are_split_under_4_mb(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport, max_buffer=1_000, flush_at=1_000)
    for i in range(100):
        acp.report_run("success", full_input="i" * 99_000, full_output="o" * 99_000, dedupe_key=f"big-{i}")
    acp.stop()
    requests = transport.to("/api/report")
    assert len(requests) > 1
    assert all(len(r.body or b"") < MAX_BATCH_BYTES + 1_000 for r in requests)
    keys = [run["dedupe_key"] for body in transport.bodies("/api/report") for run in body["runs"]]
    assert keys == [f"big-{i}" for i in range(100)]


def test_a_failed_chunk_is_dropped_and_the_rest_still_sent(
    make_client: ClientFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent_control_panel._reporter.jitter_backoff", lambda _attempt: 0.0)
    calls: list[int] = []

    def first_chunk_fails(req: Request) -> Response:
        calls.append(len(json.loads(req.body or b"{}")["runs"]))
        return Response(500) if len(calls) <= 5 else Response(200, b"{}")

    acp = make_client(transport=FakeTransport(first_chunk_fails), max_buffer=1_000, flush_at=1_000)
    for _ in range(60):
        acp.report_run("success", full_output="o" * 99_000)
    acp.stop()
    assert len(calls) > 5
    assert acp.dropped_runs == calls[0]


def test_chunking_keeps_order_and_puts_an_oversized_row_alone() -> None:
    rows = ["a" * 10, "b" * 50, "c" * 10, "d" * 10]
    assert chunk_by_size(rows, 30) == [["a" * 10], ["b" * 50], ["c" * 10, "d" * 10]]
    assert chunk_by_size([], 30) == []


# Agent framework results, wrapped as they are (no adapter).
def _graph_state() -> dict[str, Any]:
    """LangGraph's graph.invoke() result: AI messages carry usage_metadata, tool messages do not."""
    return {
        "messages": [
            SimpleNamespace(type="human", content="Where is order 42?"),
            SimpleNamespace(
                type="ai",
                content="",
                tool_calls=[{"name": "order_status"}],
                usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            ),
            SimpleNamespace(type="tool", content="Order 42 ships tomorrow.", tool_call_id="c1"),
            SimpleNamespace(
                type="ai",
                content="Order 42 ships tomorrow.",
                usage_metadata={"input_tokens": 30, "output_tokens": 8, "total_tokens": 38},
            ),
        ]
    }


def test_openai_agents_run_result_text_and_tokens() -> None:
    from agent_control_panel import extract_tokens

    run = SimpleNamespace(
        final_output="Order 42 ships tomorrow.",
        context_wrapper=SimpleNamespace(
            usage=SimpleNamespace(requests=2, input_tokens=50, output_tokens=13, total_tokens=63)
        ),
    )
    assert extract_text(run) == "Order 42 ships tomorrow."
    tokens = extract_tokens(run)
    assert tokens is not None
    assert (tokens.tokens_input, tokens.tokens_output, tokens.tokens_total) == (50, 13, 63)
    assert extract_text(SimpleNamespace(final_output={"order_id": "42"})) == '{"order_id": "42"}'


def test_langgraph_state_text_and_summed_tokens() -> None:
    from agent_control_panel import extract_tokens

    state = _graph_state()
    assert extract_text(state) == "Order 42 ships tomorrow."
    tokens = extract_tokens(state)
    assert tokens is not None
    assert (tokens.tokens_input, tokens.tokens_output, tokens.tokens_total) == (50, 13, 63)


def test_crewai_output_text_and_tokens() -> None:
    from agent_control_panel import extract_tokens

    usage = SimpleNamespace(total_tokens=63, prompt_tokens=50, completion_tokens=13, successful_requests=2)
    output = SimpleNamespace(raw="Order 42 ships tomorrow.", token_usage=usage, tasks_output=[])
    assert extract_text(output) == "Order 42 ships tomorrow."
    tokens = extract_tokens(output)
    assert tokens is not None
    assert (tokens.tokens_input, tokens.tokens_output, tokens.tokens_total) == (50, 13, 63)
    assert extract_text(SimpleNamespace(raw="no token_usage, not a CrewOutput")) is None


def test_single_langchain_message_and_unrelated_shapes() -> None:
    from agent_control_panel import extract_tokens

    message = SimpleNamespace(
        type="ai",
        content=[{"type": "text", "text": "Hi "}, {"type": "text", "text": "there"}],
        usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
    )
    assert extract_text({"messages": [message]}) == "Hi there"
    assert (
        extract_text(SimpleNamespace(content="plain", usage_metadata={"input_tokens": 1, "output_tokens": 1}))
        == "plain"
    )
    tokens = extract_tokens(message)
    assert tokens is not None
    assert tokens.tokens_total == 6
    assert extract_text({"messages": []}) is None
    assert extract_tokens({"messages": [SimpleNamespace(type="human", content="x")]}) is None
    assert extract_text(SimpleNamespace(content="no usage metadata")) is None
