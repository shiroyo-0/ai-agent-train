"""HTTP/API request tool."""

from typing import Any

import httpx

from ai_agent.core import ToolResult
from ai_agent.tools.base import BaseTool, Permission, ToolMetadata


class HttpTool(BaseTool):
    """Make HTTP requests to APIs."""

    def __init__(self) -> None:
        super().__init__(ToolMetadata(
            name="http_request",
            description="Make HTTP requests (GET, POST, PUT, DELETE). Useful for API interactions.",
            parameters={
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"], "description": "HTTP method"},
                    "url": {"type": "string", "description": "Request URL"},
                    "headers": {"type": "object", "description": "Request headers (optional)"},
                    "body": {"type": "string", "description": "Request body (optional)"},
                },
                "required": ["method", "url"],
            },
            permissions=[Permission.NETWORK],
            timeout=30,
        ))

    async def _run(self, method: str, url: str, headers: dict[str, str] | None = None, body: str | None = None, **kwargs: Any) -> ToolResult:
        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.request(
                method=method, url=url, headers=headers, content=body
            )
            output = f"Status: {response.status_code}\nHeaders: {dict(response.headers)}\n\nBody:\n{response.text[:10000]}"
            return ToolResult(
                tool_call_id="", output=output, success=response.is_success,
                metadata={"status_code": response.status_code},
            )
