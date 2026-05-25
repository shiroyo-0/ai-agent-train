"""Base protocols and types for the agent system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    output: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentState(BaseModel):
    """Serializable agent state."""
    session_id: str
    messages: list[dict[str, Any]] = []
    task: str = ""
    plan: list[str] = []
    current_step: int = 0
    iterations: int = 0
    status: TaskStatus = TaskStatus.PENDING
    context: dict[str, Any] = {}
    errors: list[str] = []


@runtime_checkable
class Tool(Protocol):
    """Protocol for tools."""
    name: str
    description: str

    async def execute(self, **kwargs: Any) -> ToolResult: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> Message: ...

    async def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> Any: ...


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for memory backends."""

    async def store(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None: ...
    async def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]: ...
    async def delete(self, key: str) -> None: ...


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, name: str, role: str) -> None:
        self.name = name
        self.role = role
        self.state = AgentState(session_id="")

    @abstractmethod
    async def execute(self, task: str, context: dict[str, Any] | None = None) -> Any:
        """Execute a task."""
        ...

    @abstractmethod
    async def plan(self, task: str) -> list[str]:
        """Create execution plan."""
        ...
