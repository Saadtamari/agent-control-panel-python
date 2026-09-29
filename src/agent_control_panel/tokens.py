"""Token counts from the response objects of common LLM clients."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    tokens_input: int
    tokens_output: int
    tokens_total: int


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(v: Any) -> int:
    if _is_number(v) and math.isfinite(v) and v > 0:
        return int(v)
    return 0


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _is_object(obj: Any) -> bool:
    return obj is not None and not isinstance(obj, (str, bytes, bytearray, int, float, bool, list, tuple))


def _has(obj: Any, *keys: str) -> bool:
    return any(_is_number(_get(obj, k)) for k in keys)


def _usage(tokens_input: int, tokens_output: int, total: Any = None) -> TokenUsage:
    return TokenUsage(tokens_input, tokens_output, _num(total) or tokens_input + tokens_output)


def _from_usage(u: Any) -> TokenUsage | None:
    if _has(u, "prompt_tokens", "completion_tokens"):
        return _usage(
            _num(_get(u, "prompt_tokens")), _num(_get(u, "completion_tokens")), _get(u, "total_tokens")
        )
    if _has(u, "input_tokens", "output_tokens"):
        # Anthropic's input_tokens excludes prompt-cache tokens; they are input too.
        tokens_in = (
            _num(_get(u, "input_tokens"))
            + _num(_get(u, "cache_creation_input_tokens"))
            + _num(_get(u, "cache_read_input_tokens"))
        )
        return _usage(tokens_in, _num(_get(u, "output_tokens")), _get(u, "total_tokens"))
    if _has(u, "inputTokens", "outputTokens"):
        return _usage(_num(_get(u, "inputTokens")), _num(_get(u, "outputTokens")), _get(u, "totalTokens"))
    if _has(u, "promptTokens", "completionTokens"):
        return _usage(
            _num(_get(u, "promptTokens")), _num(_get(u, "completionTokens")), _get(u, "totalTokens")
        )
    return None


def _from_gemini(meta: Any) -> TokenUsage | None:
    if _is_object(meta) and _has(meta, "promptTokenCount", "candidatesTokenCount"):
        return _usage(
            _num(_get(meta, "promptTokenCount")),
            _num(_get(meta, "candidatesTokenCount")),
            _get(meta, "totalTokenCount"),
        )
    if _is_object(meta) and _has(meta, "prompt_token_count", "candidates_token_count"):
        return _usage(
            _num(_get(meta, "prompt_token_count")),
            _num(_get(meta, "candidates_token_count")),
            _get(meta, "total_token_count"),
        )
    return None


def extract_tokens(result: Any) -> TokenUsage | None:
    """Token counts from an LLM response (a dict or an SDK object), or None when none are recognisable.

    - OpenAI Chat Completions: ``usage.prompt_tokens`` / ``completion_tokens`` / ``total_tokens``
    - OpenAI Responses API and Anthropic Messages: ``usage.input_tokens`` / ``output_tokens``; Anthropic's
      prompt-cache tokens (``cache_creation_input_tokens``, ``cache_read_input_tokens``) count as input
    - Google Gemini: ``usage_metadata`` (Python SDK) or ``usageMetadata`` (REST JSON), also under ``response``
    - camelCase ``usage.inputTokens`` / ``promptTokens`` shapes (as sent by the Vercel AI SDK)

    Never raises; report tokens yourself with ``report_run`` when this returns None.
    """
    try:
        if not _is_object(result):
            return None
        u = _get(result, "usage")
        if _is_object(u):
            found = _from_usage(u)
            if found is not None:
                return found
        response = _get(result, "response")
        for meta in (
            _get(result, "usageMetadata"),
            _get(response, "usageMetadata") if _is_object(response) else None,
            _get(result, "usage_metadata"),
            _get(response, "usage_metadata") if _is_object(response) else None,
        ):
            found = _from_gemini(meta)
            if found is not None:
                return found
    except Exception:
        return None
    return None
