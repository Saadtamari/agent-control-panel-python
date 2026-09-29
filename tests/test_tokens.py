from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from agent_control_panel import TokenUsage, extract_tokens


def test_openai_and_anthropic_sdk_objects() -> None:
    openai_chat = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=None
        )
    )
    assert extract_tokens(openai_chat) == TokenUsage(10, 5, 15)

    anthropic = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_creation_input_tokens=None, cache_read_input_tokens=100
        )
    )
    assert extract_tokens(anthropic) == TokenUsage(110, 5, 115)


def test_gemini_python_sdk_snake_case() -> None:
    gemini = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=2, total_token_count=12)
    )
    assert extract_tokens(gemini) == TokenUsage(8, 2, 12)
    wrapped = SimpleNamespace(
        response=SimpleNamespace(usage_metadata={"prompt_token_count": 3, "candidates_token_count": 1})
    )
    assert extract_tokens(wrapped) == TokenUsage(3, 1, 4)


def test_dataclasses_and_mapping_subclasses() -> None:
    @dataclass
    class Usage:
        prompt_tokens: int
        completion_tokens: int

    @dataclass
    class Result:
        usage: Usage

    assert extract_tokens(Result(Usage(2, 3))) == TokenUsage(2, 3, 5)
    assert extract_tokens(OrderedDict(usage=OrderedDict(input_tokens=1, output_tokens=1))) == TokenUsage(
        1, 1, 2
    )


def test_booleans_floats_and_junk() -> None:
    assert extract_tokens({"usage": {"prompt_tokens": True, "completion_tokens": True}}) is None
    assert extract_tokens({"usage": {"prompt_tokens": 2.9, "completion_tokens": float("nan")}}) == TokenUsage(
        2, 0, 2
    )
    assert extract_tokens({"usage": {"input_tokens": float("inf"), "output_tokens": 1}}) == TokenUsage(
        0, 1, 1
    )
    for junk in ([{"usage": {"prompt_tokens": 1}}], b"bytes", (1, 2), 3.5, True, object()):
        assert extract_tokens(junk) is None


def test_an_object_whose_attributes_raise_is_ignored() -> None:
    class Exploding:
        @property
        def usage(self) -> Any:
            raise RuntimeError("lazy attribute failed")

    assert extract_tokens(Exploding()) is None
