"""Git tools for repository operations."""

import asyncio
from typing import Any

from ai_agent.core import ToolResult
from ai_agent.tools.base import BaseTool, Permission, ToolMetadata


class GitTool(BaseTool):
    """Git operations - status, diff, log, commit, branch management."""

    def __init__(self) -> None:
        super().__init__(ToolMetadata(
            name="git",
            description="Execute git operations: status, diff, log, add, commit, branch, checkout, stash.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["status", "diff", "log", "add", "commit", "branch", "checkout", "stash", "show", "blame"],
                        "description": "Git operation to perform",
                    },
                    "args": {"type": "string", "description": "Additional arguments for the git command"},
                    "working_dir": {"type": "string", "description": "Repository directory"},
                },
                "required": ["operation"],
            },
            permissions=[Permission.READ, Permission.WRITE],
        ))

    async def _run(self, operation: str, args: str = "", working_dir: str = ".", **kwargs: Any) -> ToolResult:
        safe_ops = {"status", "diff", "log", "show", "blame", "branch"}
        if operation not in safe_ops and operation in {"add", "commit", "checkout", "stash"}:
            pass  # allowed but tracked

        cmd = f"git {operation} {args}".strip()
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode(errors="replace")
        if stderr:
            output += f"\n{stderr.decode(errors='replace')}"

        if len(output) > 50000:
            output = output[:50000] + "\n... [truncated]"

        return ToolResult(
            tool_call_id="",
            output=output or "(no output)",
            success=proc.returncode == 0,
        )
