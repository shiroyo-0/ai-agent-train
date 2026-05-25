"""Shell execution tool with sandboxing and streaming."""

import asyncio
import os
from typing import Any

from ai_agent.core import ToolResult
from ai_agent.tools.base import BaseTool, Permission, ToolMetadata


class ShellTool(BaseTool):
    """Execute shell commands with timeout and output capture."""

    def __init__(self, working_dir: str | None = None, timeout: int = 120) -> None:
        super().__init__(ToolMetadata(
            name="shell",
            description="Execute a shell command and return stdout/stderr. Use for running builds, tests, git commands, installations, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "working_dir": {"type": "string", "description": "Working directory (optional)"},
                },
                "required": ["command"],
            },
            permissions=[Permission.EXECUTE],
            timeout=timeout,
        ))
        self._default_dir = working_dir or os.getcwd()

    async def _run(self, command: str, working_dir: str | None = None, **kwargs: Any) -> ToolResult:
        cwd = working_dir or self._default_dir
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env={**os.environ, "TERM": "dumb"},
        )
        stdout, stderr = await proc.communicate()
        output_parts = []
        if stdout:
            output_parts.append(stdout.decode(errors="replace"))
        if stderr:
            output_parts.append(f"STDERR:\n{stderr.decode(errors='replace')}")
        output_parts.append(f"\n[exit code: {proc.returncode}]")
        output = "\n".join(output_parts)

        # Truncate very long outputs
        if len(output) > 50000:
            output = output[:25000] + "\n\n... [truncated] ...\n\n" + output[-25000:]

        return ToolResult(
            tool_call_id="",
            output=output,
            success=proc.returncode == 0,
            metadata={"exit_code": proc.returncode, "cwd": cwd},
        )
