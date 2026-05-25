"""Tool system exports and default tool registration."""

from ai_agent.tools.base import BaseTool, Permission, ToolMetadata, ToolRegistry
from ai_agent.tools.code_search import CodeSearchTool
from ai_agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, SearchFilesTool, WriteFileTool
from ai_agent.tools.git import GitTool
from ai_agent.tools.http import HttpTool
from ai_agent.tools.sandbox import DockerSandboxTool, PythonExecTool
from ai_agent.tools.shell import ShellTool


def create_default_registry(working_dir: str | None = None) -> ToolRegistry:
    """Create a registry with all default tools."""
    registry = ToolRegistry()
    registry.register(ShellTool(working_dir=working_dir))
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ListDirTool())
    registry.register(SearchFilesTool())
    registry.register(GitTool())
    registry.register(CodeSearchTool())
    registry.register(PythonExecTool())
    registry.register(DockerSandboxTool())
    registry.register(HttpTool())
    return registry


__all__ = [
    "BaseTool", "CodeSearchTool", "DockerSandboxTool", "EditFileTool",
    "GitTool", "HttpTool", "ListDirTool", "Permission", "PythonExecTool",
    "ReadFileTool", "SearchFilesTool", "ShellTool", "ToolMetadata",
    "ToolRegistry", "WriteFileTool", "create_default_registry",
]
