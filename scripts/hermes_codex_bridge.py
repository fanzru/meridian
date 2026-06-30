#!/usr/bin/env python3
"""Bridge Meridian's chat-completions loop to Hermes Codex OAuth.

Reads a JSON request on stdin with OpenAI-chat-like fields:
  {
    "model": str,
    "messages": [...],
    "tools": [...],
    "tool_choice": "auto" | "required" | "none",
    "max_tokens": int | null,
    "session_id": str | null
  }

Resolves fresh Codex OAuth credentials via Hermes, calls the ChatGPT Codex
Responses API through Hermes' own adapter logic, and prints a compact
chat-completions-like JSON response to stdout:
  {
    "choices": [{"message": {"role": "assistant", "content": str|None,
                                "tool_calls": [...]}}]
  }

No secrets are printed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _add_hermes_repo_to_path() -> None:
    explicit = os.getenv("HERMES_AGENT_CODEBASE", "").strip()
    candidates = [
        explicit,
        str(Path.home() / ".hermes" / "hermes-agent"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate).expanduser()
        if (p / "hermes_cli" / "auth.py").is_file():
            sys.path.insert(0, str(p))
            return
    raise RuntimeError(
        "Hermes Agent codebase not found. Set HERMES_AGENT_CODEBASE or install Hermes under ~/.hermes/hermes-agent."
    )


_add_hermes_repo_to_path()

from hermes_cli.auth import resolve_codex_runtime_credentials  # noqa: E402
from agent.transports.codex import ResponsesApiTransport  # noqa: E402
from openai import OpenAI  # noqa: E402


def _load_request() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise RuntimeError("No JSON request received on stdin.")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("Request JSON must be an object.")
    return data


def _combine_tool_id(call_id: str | None, response_item_id: str | None) -> str | None:
    call_id = (call_id or "").strip() or None
    response_item_id = (response_item_id or "").strip() or None
    if call_id and response_item_id:
        return f"{call_id}|{response_item_id}"
    return response_item_id or call_id


def _translate_tool_choice(raw: Any, tools: List[Dict[str, Any]]) -> Any:
    if raw == "none":
        return "none"
    if raw == "required" and tools:
        allowed = []
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str) and name.strip():
                allowed.append({"type": "function", "name": name.strip()})
        if allowed:
            return {"type": "allowed_tools", "mode": "required", "tools": allowed}
    return "auto"


def _collect_stream_result(stream) -> Dict[str, Any]:
    content_chunks: List[str] = []
    tool_state: Dict[str, Dict[str, Any]] = {}

    for event in stream:
        etype = getattr(event, "type", "")
        if etype == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            if isinstance(delta, str):
                content_chunks.append(delta)
            continue

        if etype == "response.output_item.added":
            item = getattr(event, "item", None)
            if getattr(item, "type", None) == "function_call":
                item_id = getattr(item, "id", None) or getattr(event, "item_id", None)
                if isinstance(item_id, str) and item_id:
                    tool_state[item_id] = {
                        "id": _combine_tool_id(getattr(item, "call_id", None), item_id),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", "") or "",
                            "arguments": getattr(item, "arguments", "") or "",
                        },
                    }
            continue

        if etype == "response.function_call_arguments.delta":
            item_id = getattr(event, "item_id", None)
            delta = getattr(event, "delta", None)
            if isinstance(item_id, str) and item_id and isinstance(delta, str):
                tc = tool_state.setdefault(item_id, {
                    "id": _combine_tool_id(None, item_id),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                tc["function"]["arguments"] += delta
            continue

        if etype in {"response.function_call_arguments.done", "response.output_item.done"}:
            item = getattr(event, "item", None)
            item_id = getattr(item, "id", None) if item is not None else getattr(event, "item_id", None)
            item_type = getattr(item, "type", None) if item is not None else None
            if etype == "response.output_item.done" and item_type != "function_call":
                continue
            if isinstance(item_id, str) and item_id:
                tc = tool_state.setdefault(item_id, {
                    "id": _combine_tool_id(None, item_id),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if item is not None and item_type == "function_call":
                    tc["id"] = _combine_tool_id(getattr(item, "call_id", None), item_id)
                    name = getattr(item, "name", None)
                    arguments = getattr(item, "arguments", None)
                    if isinstance(name, str) and name:
                        tc["function"]["name"] = name
                    if isinstance(arguments, str):
                        tc["function"]["arguments"] = arguments
                else:
                    arguments = getattr(event, "arguments", None)
                    if isinstance(arguments, str):
                        tc["function"]["arguments"] = arguments
            continue

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_chunks) or None,
    }
    tool_calls = list(tool_state.values())
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {"choices": [{"message": message}]}


def main() -> int:
    try:
        req = _load_request()
        creds = resolve_codex_runtime_credentials()
        transport = ResponsesApiTransport()

        model = str(req.get("model") or "gpt-5.4-mini").strip()
        messages = req.get("messages")
        tools = req.get("tools") or []
        tool_choice = req.get("tool_choice")
        max_tokens = req.get("max_tokens")
        session_id = str(req.get("session_id") or "meridian-hermes-codex").strip()

        if not isinstance(messages, list) or not messages:
            raise RuntimeError("messages must be a non-empty array")
        if not isinstance(tools, list):
            raise RuntimeError("tools must be an array")

        kwargs = transport.build_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            provider="openai-codex",
            base_url=creds["base_url"],
            is_codex_backend=True,
            max_tokens=max_tokens,
            session_id=session_id,
        )
        translated_tool_choice = _translate_tool_choice(tool_choice, tools)
        if translated_tool_choice == "none":
            kwargs.pop("tools", None)
            kwargs["tool_choice"] = "none"
        elif translated_tool_choice != "auto":
            kwargs["tool_choice"] = translated_tool_choice

        kwargs = transport.preflight_kwargs(kwargs, allow_stream=True)

        client = OpenAI(base_url=creds["base_url"], api_key=creds["api_key"], timeout=300)
        with client.responses.stream(**kwargs) as stream:
            result = _collect_stream_result(stream)

        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
