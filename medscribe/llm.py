"""LLM interface.

The agent depends only on `chat()`, so tests inject a scripted fake and the OpenAI client is
created lazily. Importing this package never needs an API key.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import config


class MissingAPIKey(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class Turn:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def as_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": tc.raw_arguments or json.dumps(tc.arguments)}}
                for tc in self.tool_calls
            ]
        return msg


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int | None = None, json_mode: bool = False) -> Turn: ...


class OpenAIChat:
    def __init__(self, model: str | None = None, api_key: str | None = None, client: Any = None):
        self.model = model or config.MODEL
        self._api_key = api_key
        self._client = client

    @property
    def client(self):
        if self._client is None:
            key = self._api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise MissingAPIKey("OPENAI_API_KEY is not set. Put it in .env or a Kubernetes secret.")
            from openai import OpenAI
            self._client = OpenAI(api_key=key, max_retries=3, timeout=90)
        return self._client

    def chat(self, messages, tools=None, max_tokens=None, json_mode=False) -> Turn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens or config.MAX_OUTPUT_TOKENS,
        }
        if config.TEMPERATURE is not None:
            kwargs["temperature"] = config.TEMPERATURE
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            args: dict[str, Any] = {}
            err = None
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    err = "tool arguments must be a JSON object"
            except json.JSONDecodeError as e:
                err = f"tool arguments were not valid JSON: {e}"
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args, raw_arguments=raw,
                                  parse_error=err))
        usage = {}
        if getattr(resp, "usage", None):
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        return Turn(content=msg.content, tool_calls=calls, usage=usage)
