"""LLM provider system using LiteLLM for multi-provider support."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import litellm
import tiktoken

from ai_agent.core import Message, Role, ToolCall, get_logger, get_settings

logger = get_logger(__name__)

litellm.drop_params = True
litellm.set_verbose = False


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


@dataclass
class CompletionResponse:
    message: Message
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    finish_reason: str = ""


@dataclass
class StreamChunk:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None


def _messages_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    result = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role.value, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        result.append(d)
    return result


def _parse_tool_calls(raw: list[Any]) -> list[ToolCall]:
    import json
    calls = []
    for tc in raw:
        fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
        name = fn.name if hasattr(fn, "name") else fn.get("name", "")
        args_str = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {"raw": args_str}
        calls.append(ToolCall(
            id=tc.id if hasattr(tc, "id") else tc.get("id", ""),
            name=name,
            arguments=args,
        ))
    return calls


class LLMClient:
    """Unified LLM client supporting all providers via LiteLLM."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.default_model
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens
        self._fallback_model = settings.fallback_model
        self._total_usage = TokenUsage()

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send completion request with automatic fallback."""
        target_model = model or self.model
        params: dict[str, Any] = {
            "model": target_model,
            "messages": _messages_to_dicts(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        try:
            response = await litellm.acompletion(**params)
        except Exception as e:
            logger.warning("primary_model_failed", model=target_model, error=str(e))
            if target_model != self._fallback_model:
                params["model"] = self._fallback_model
                response = await litellm.acompletion(**params)
            else:
                raise

        choice = response.choices[0]
        msg_data = choice.message

        tool_calls_raw = []
        if hasattr(msg_data, "tool_calls") and msg_data.tool_calls:
            tool_calls_raw = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg_data.tool_calls
            ]

        message = Message(
            role=Role.ASSISTANT,
            content=msg_data.content or "",
            tool_calls=tool_calls_raw,
        )

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            cost=litellm.completion_cost(response) if response.usage else 0.0,
        )
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens
        self._total_usage.total_tokens += usage.total_tokens
        self._total_usage.cost += usage.cost

        return CompletionResponse(
            message=message,
            usage=usage,
            model=response.model or target_model,
            finish_reason=choice.finish_reason or "",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion response."""
        target_model = model or self.model
        params: dict[str, Any] = {
            "model": target_model,
            "messages": _messages_to_dicts(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = await litellm.acompletion(**params)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            yield StreamChunk(
                content=delta.content or "",
                tool_calls=[
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in (delta.tool_calls or [])
                ] if hasattr(delta, "tool_calls") and delta.tool_calls else [],
                finish_reason=chunk.choices[0].finish_reason,
            )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens for text."""
        try:
            enc = tiktoken.encoding_for_model(model or self.model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    @property
    def total_usage(self) -> TokenUsage:
        return self._total_usage

    def reset_usage(self) -> None:
        self._total_usage = TokenUsage()
