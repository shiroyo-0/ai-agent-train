"""Python code execution sandbox."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from ai_agent.core import ToolResult
from ai_agent.tools.base import BaseTool, Permission, ToolMetadata


class PythonExecTool(BaseTool):
    """Execute Python code in an isolated subprocess."""

    def __init__(self, timeout: int = 30) -> None:
        super().__init__(ToolMetadata(
            name="python_exec",
            description="Execute Python code and return the output. Runs in a subprocess for isolation.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Execution timeout in seconds (default: 30)"},
                },
                "required": ["code"],
            },
            permissions=[Permission.EXECUTE],
            timeout=timeout,
        ))

    async def _run(self, code: str, timeout: int = 30, **kwargs: Any) -> ToolResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode(errors="replace")
            if stderr:
                output += f"\nSTDERR:\n{stderr.decode(errors='replace')}"
            output += f"\n[exit code: {proc.returncode}]"
            return ToolResult(tool_call_id="", output=output, success=proc.returncode == 0)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(tool_call_id="", output=f"Execution timed out after {timeout}s", success=False)
        finally:
            Path(script_path).unlink(missing_ok=True)


class DockerSandboxTool(BaseTool):
    """Execute commands in a Docker container for full isolation."""

    def __init__(self) -> None:
        super().__init__(ToolMetadata(
            name="sandbox",
            description="Execute a command in an isolated Docker container. Use for untrusted code or dangerous operations.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute in the sandbox"},
                    "image": {"type": "string", "description": "Docker image (default: python:3.12-slim)"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)"},
                },
                "required": ["command"],
            },
            permissions=[Permission.EXECUTE, Permission.DANGEROUS],
            timeout=120,
            requires_confirmation=True,
        ))

    async def _run(self, command: str, image: str = "python:3.12-slim", timeout: int = 60, **kwargs: Any) -> ToolResult:
        docker_cmd = (
            f"docker run --rm --network none --memory 512m --cpus 1 "
            f"--timeout {timeout} {image} sh -c {_shell_quote(command)}"
        )
        proc = await asyncio.create_subprocess_shell(
            docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 10)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(tool_call_id="", output="Sandbox execution timed out", success=False)

        output = stdout.decode(errors="replace")
        if stderr:
            output += f"\nSTDERR:\n{stderr.decode(errors='replace')}"
        return ToolResult(tool_call_id="", output=output, success=proc.returncode == 0)


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"
