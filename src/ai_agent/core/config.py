"""Core configuration using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_AGENT_", env_file=".env", extra="ignore")

    # Model
    default_model: str = "anthropic/claude-sonnet-4-20250514"
    fallback_model: str = "ollama/llama3"
    temperature: float = 0.1
    max_tokens: int = 8192

    # Agent
    max_iterations: int = 50
    timeout: int = 300
    reasoning_depth: Literal["shallow", "standard", "deep"] = "standard"

    # Security
    sandbox_enabled: bool = True
    allow_shell: bool = True
    allow_network: bool = True
    audit_log: bool = True

    # Paths
    data_dir: Path = Path.home() / ".ai-agent"
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://agent:agent@localhost:5432/ai_agent"
    redis_url: str = "redis://localhost:6379/0"

    # Vector Store
    chroma_host: str = "localhost"
    chroma_port: int = 8100
    embedding_model: str = "all-MiniLM-L6-v2"

    # Provider Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8000"

    def ensure_dirs(self) -> None:
        """Create required directories."""
        for sub in ["sessions", "logs", "memory", "embeddings", "cache"]:
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
