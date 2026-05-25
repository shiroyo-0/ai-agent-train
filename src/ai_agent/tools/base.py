"""Tool system base - registry, permissions, and execution framework."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_agent.core import Event, ToolResult, get_event_bus, get_logger, get_settings

logger = get_logger(__name__)


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    DANGEROUS = "dangerous"


@dataclass
class ToolMetadata:
    name: str
    description: str
    parameters: dict[str, Any]
    permissions: list[Permission] = field(default_factory=list)
    timeout: int = 60
    requires_confirmation: bool = False


class BaseTool(ABC):
    """Base class for all tools with permission checking and audit logging."""

    def __init__(self, metadata: ToolMetadata) -> None:
        self._metadata = metadata
        self._event_bus = get_event_bus()
        self._settings = get_settings()

    @property
    def name(self) -> str:
        return self._metadata.name

    @property
    def description(self) -> str:
        return self._metadata.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._metadata.parameters

    @property
    def permissions(self) -> list[Permission]:
        return self._metadata.permissions

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute with permission check, timeout, and audit."""
        if not self._check_permissions():
            return ToolResult(
                tool_call_id="", output=f"Permission denied for tool '{self.name}'", success=False
            )

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._run(**kwargs), timeout=self._metadata.timeout
            )
            elapsed = time.monotonic() - start
            if self._settings.audit_log:
                await self._audit(kwargs, result, elapsed)
            return result
        except asyncio.TimeoutError:
            return ToolResult(tool_call_id="", output=f"Tool '{self.name}' timed out after {self._metadata.timeout}s", success=False)
        except Exception as e:
            return ToolResult(tool_call_id="", output=f"Tool error: {e}", success=False)

    @abstractmethod
    async def _run(self, **kwargs: Any) -> ToolResult:
        """Actual tool implementation."""
        ...

    def _check_permissions(self) -> bool:
        settings = self._settings
        if Permission.EXECUTE in self.permissions and not settings.allow_shell:
            return False
        if Permission.NETWORK in self.permissions and not settings.allow_network:
            return False
        return True

    async def _audit(self, args: dict, result: ToolResult, elapsed: float) -> None:
        await self._event_bus.publish(Event(
            type="tool.audit",
            data={"tool": self.name, "args": args, "success": result.success, "elapsed": elapsed},
        ))


class ToolRegistry:
    """Central registry for all tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    @property
    def as_dict(self) -> dict[str, BaseTool]:
        return self._tools.copy()
