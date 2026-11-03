"""Bring tool-call arguments to the types the tool schema declares.

Different vLLM tool-call parsers return argument values differently: the GLM parser JSON-decodes every
value, so a plan's `steps` can arrive as a JSON *string* (crashing plan validation) and a Mongo `pipeline`
that the schema declares as a string arrives as a list (rejected by the server). This module converts
values only when their type contradicts the schema, and reports every conversion so it is recorded with
the model call (`coerced` in the llm_call event's tool_calls). Correctly typed arguments are untouched.
"""
from __future__ import annotations

import json
from typing import Any


def _coerce(v: Any, schema: dict | None, path: str, notes: list[str]) -> Any:
    if not isinstance(schema, dict):
        return v
    t = schema.get("type")
    if t == "array":
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError:
                return v
            if isinstance(parsed, list):
                notes.append(f"{path}: JSON string -> array")
                v = parsed
        if isinstance(v, list) and isinstance(schema.get("items"), dict):
            v = [_coerce(x, schema["items"], f"{path}[{i}]", notes) for i, x in enumerate(v)]
        return v
    if t == "object":
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError:
                return v
            if isinstance(parsed, dict):
                notes.append(f"{path}: JSON string -> object")
                v = parsed
        if isinstance(v, dict) and isinstance(schema.get("properties"), dict):
            v = {k: _coerce(x, schema["properties"].get(k), f"{path}.{k}", notes) for k, x in v.items()}
        return v
    if t == "string" and v is not None and not isinstance(v, str):
        notes.append(f"{path}: {type(v).__name__} -> string")
        return json.dumps(v) if isinstance(v, (list, dict)) else str(v)
    if t in ("integer", "number") and isinstance(v, str):
        try:
            out = int(v) if t == "integer" else float(v)
        except ValueError:
            return v
        notes.append(f"{path}: string -> {t}")
        return out
    return v


def coerce_args(args: dict, schema: dict | None) -> tuple[dict, list[str]]:
    """Return (arguments with schema-conforming types, list of conversions made)."""
    notes: list[str] = []
    if not isinstance(args, dict) or not isinstance(schema, dict):
        return args, notes
    props = schema.get("properties") or {}
    out = {k: _coerce(v, props.get(k), k, notes) for k, v in args.items()}
    return out, notes
