"""Run content: the answer text of LLM responses, and cutting content to what ACP stores."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

MAX_FULL_CHARS = 100_000
"""What ACP stores per full input/output field, in UTF-16 code units (JavaScript string length)."""
MAX_SUMMARY_CHARS = 2_000
"""What ACP stores per summary field."""


def _read(obj: Any, key: str) -> Any:
    """A key or plain attribute. Never calls anything: a callable value (a method) counts as absent."""
    try:
        value = obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)
    except Exception:
        return None
    return None if callable(value) else value


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_object(value: Any) -> bool:
    return value is not None and not isinstance(value, (str, bytes, bytearray, int, float, bool, list, tuple))


def _join_text(parts: Any) -> str | None:
    if not isinstance(parts, (list, tuple)):
        return None
    texts = [p if isinstance(p, str) else _text(_read(p, "text")) if _is_object(p) else None for p in parts]
    found = [t for t in texts if t is not None]
    return "".join(found) if found else None


def _extract(result: Any) -> str | None:
    if isinstance(result, str):
        return result or None
    if not _is_object(result):
        return None
    output_text = _text(_read(result, "output_text"))
    if output_text:
        return output_text
    choices = _read(result, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        message = _read(choices[0], "message") if _is_object(choices[0]) else None
        content = _read(message, "content") if _is_object(message) else None
        text = _text(content) or _join_text(content)
        if text:
            return text
    content = _read(result, "content")
    if isinstance(content, (list, tuple)):
        blocks = [b for b in content if _is_object(b) and _read(b, "type") == "text"]
        text = _join_text(blocks)
        if text:
            return text
    response = _read(result, "response")
    candidates = _read(result, "candidates")
    if not isinstance(candidates, (list, tuple)) and _is_object(response):
        candidates = _read(response, "candidates")
    if isinstance(candidates, (list, tuple)) and candidates and _is_object(candidates[0]):
        body = _read(candidates[0], "content")
        text = _join_text(_read(body, "parts")) if _is_object(body) else None
        if text:
            return text
    return _text(_read(result, "text"))


def extract_text(result: Any) -> str | None:
    """The answer text of a common LLM client response (a dict or an SDK object), or None.

    - a string result is the text itself
    - OpenAI Chat Completions: ``choices[0].message.content`` (a string or text parts)
    - OpenAI Responses API: ``output_text``
    - Anthropic Messages: ``content`` blocks of type ``text``
    - Google Gemini: ``candidates[0].content.parts[].text`` (also under ``response``)
    - ``text`` (Vercel AI SDK shape)

    Only reads keys and attributes; never calls a method on the result. Never raises.
    """
    try:
        return _extract(result)
    except Exception:
        return None


def cut_utf16(text: str, limit: int) -> tuple[str, bool]:
    """Cut ``text`` to ``limit`` UTF-16 code units (how JavaScript and the ACP server count length)."""
    if len(text) * 2 <= limit or len(text.encode("utf-16-le")) // 2 <= limit:
        return text, False
    units = 0
    for index, char in enumerate(text):
        units += 2 if ord(char) > 0xFFFF else 1
        if units > limit:
            return text[:index], True
    return text, False


def clamp_full(value: Any) -> tuple[str | None, bool]:
    """Serialise a full-content field like ``JSON.stringify`` and cut it to ACP's limit.

    A value that cannot be serialised (circular, NaN) is left out: (None, False).
    """
    if value is None:
        return None, False
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=str)
        except (TypeError, ValueError, RecursionError):
            return None, False
    return cut_utf16(text, MAX_FULL_CHARS)


def clamp_summary(value: str | None) -> str | None:
    return None if value is None else cut_utf16(value, MAX_SUMMARY_CHARS)[0]


def run_row(fields: dict[str, Any], capture_content: bool) -> dict[str, Any]:
    """The wire row for one run: content cut (and flagged) or, with capture off, left out; None dropped."""
    content = ("input_summary", "output_summary", "user_input")
    row = {k: v for k, v in fields.items() if k not in ("full_input", "full_output")}
    if not capture_content:
        for key in content:
            row.pop(key, None)
    full_in, in_cut = clamp_full(fields.get("full_input")) if capture_content else (None, False)
    full_out, out_cut = clamp_full(fields.get("full_output")) if capture_content else (None, False)
    row.update(full_input=full_in, full_output=full_out)
    if in_cut:
        row["full_input_truncated"] = True
    if out_cut:
        row["full_output_truncated"] = True
    return {k: v for k, v in row.items() if v is not None}
