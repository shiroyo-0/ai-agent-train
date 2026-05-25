"""Model registry and management."""

from dataclasses import dataclass
from typing import Any

from ai_agent.core import get_settings


@dataclass
class ModelInfo:
    id: str
    provider: str
    name: str
    context_window: int
    max_output: int
    supports_tools: bool = True
    supports_vision: bool = False
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


MODELS: dict[str, ModelInfo] = {
    "anthropic/claude-sonnet-4-20250514": ModelInfo(
        id="anthropic/claude-sonnet-4-20250514", provider="anthropic", name="Claude Sonnet 4",
        context_window=200000, max_output=8192, supports_tools=True, supports_vision=True,
        cost_per_1k_input=0.003, cost_per_1k_output=0.015,
    ),
    "openai/gpt-4o": ModelInfo(
        id="openai/gpt-4o", provider="openai", name="GPT-4o",
        context_window=128000, max_output=4096, supports_tools=True, supports_vision=True,
        cost_per_1k_input=0.005, cost_per_1k_output=0.015,
    ),
    "openai/gpt-4o-mini": ModelInfo(
        id="openai/gpt-4o-mini", provider="openai", name="GPT-4o Mini",
        context_window=128000, max_output=4096, supports_tools=True, supports_vision=True,
        cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
    ),
    "groq/llama3-70b-8192": ModelInfo(
        id="groq/llama3-70b-8192", provider="groq", name="Llama 3 70B",
        context_window=8192, max_output=8192, supports_tools=True,
        cost_per_1k_input=0.00059, cost_per_1k_output=0.00079,
    ),
    "ollama/llama3": ModelInfo(
        id="ollama/llama3", provider="ollama", name="Llama 3 (Local)",
        context_window=8192, max_output=4096, supports_tools=True,
    ),
    "ollama/codellama": ModelInfo(
        id="ollama/codellama", provider="ollama", name="CodeLlama (Local)",
        context_window=16384, max_output=4096, supports_tools=False,
    ),
}


def get_model_info(model_id: str) -> ModelInfo | None:
    return MODELS.get(model_id)


def list_models(provider: str | None = None) -> list[ModelInfo]:
    models = list(MODELS.values())
    if provider:
        models = [m for m in models if m.provider == provider]
    return models
