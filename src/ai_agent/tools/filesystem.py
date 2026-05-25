"""Filesystem tools - read, write, list, search files."""

import os
from pathlib import Path
from typing import Any

from ai_agent.core import ToolResult
from ai_agent.tools.base import BaseTool, Permission, ToolMetadata


class ReadFileTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(ToolMetadata(
            name="read_file",
            description="Read the contents of a file. Can read specific line ranges.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "start_line": {"type": "integer", "description": "Start line (1-indexed, optional)"},
                    "end_line": {"type": "integer", "description": "End line (1-indexed, optional)"},
                },
                "required": ["path"],
            },
            permissions=[Permission.READ],
        ))

    async def _run(self, path: str, start_line: int | None = None, end_line: int | None = None, **kwargs: Any) -> ToolResult:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return ToolResult(tool_call_id="", output=f"File not found: {path}", success=False)
        if not p.is_file():
            return ToolResult(tool_call_id="", output=f"Not a file: {path}", success=False)

        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult(tool_call_id="", output=f"Error reading file: {e}", success=False)

        if start_line or end_line:
            lines = content.splitlines(keepends=True)
            start = (start_line or 1) - 1
            end = end_line or len(lines)
            content = "".join(lines[start:end])
            header = f"[Lines {start+1}-{min(end, len(lines))} of {len(lines)}]\n"
            content = header + content

        if len(content) > 100000:
            content = content[:100000] + "\n\n... [file truncated, too large] ..."

        return ToolResult(tool_call_id="", output=content, success=True, metadata={"path": str(p)})


class WriteFileTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(ToolMetadata(
            name="write_file",
            description="Write content to a file. Creates parent directories if needed.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
            permissions=[Permission.WRITE],
        ))

    async def _run(self, path: str, content: str, **kwargs: Any) -> ToolResult:
        p = Path(path).expanduser().resolve()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(tool_call_id="", output=f"Written {len(content)} bytes to {p}", success=True)
        except Exception as e:
            return ToolResult(tool_call_id="", output=f"Error writing file: {e}", success=False)


class EditFileTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(ToolMetadata(
            name="edit_file",
            description="Edit a file by replacing a specific string with new content. Use for precise edits.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_str": {"type": "string", "description": "Exact string to find and replace"},
                    "new_str": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
            permissions=[Permission.WRITE],
        ))

    async def _run(self, path: str, old_str: str, new_str: str, **kwargs: Any) -> ToolResult:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return ToolResult(tool_call_id="", output=f"File not found: {path}", success=False)

        content = p.read_text(encoding="utf-8")
        if old_str not in content:
            return ToolResult(tool_call_id="", output=f"String not found in {path}. Make sure the old_str matches exactly.", success=False)

        count = content.count(old_str)
        if count > 1:
            return ToolResult(tool_call_id="", output=f"Found {count} occurrences. Please provide a more specific string.", success=False)

        new_content = content.replace(old_str, new_str, 1)
        p.write_text(new_content, encoding="utf-8")
        return ToolResult(tool_call_id="", output=f"Edited {p}: replaced 1 occurrence", success=True)


class ListDirTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(ToolMetadata(
            name="list_dir",
            description="List directory contents with file sizes and types.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "recursive": {"type": "boolean", "description": "List recursively (default: false)"},
                    "max_depth": {"type": "integer", "description": "Max recursion depth (default: 2)"},
                },
                "required": ["path"],
            },
            permissions=[Permission.READ],
        ))

    async def _run(self, path: str, recursive: bool = False, max_depth: int = 2, **kwargs: Any) -> ToolResult:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return ToolResult(tool_call_id="", output=f"Directory not found: {path}", success=False)
        if not p.is_dir():
            return ToolResult(tool_call_id="", output=f"Not a directory: {path}", success=False)

        ignore = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}
        entries = []

        def _list(dir_path: Path, depth: int = 0) -> None:
            if depth > max_depth:
                return
            try:
                items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            except PermissionError:
                return
            for item in items:
                if item.name in ignore:
                    continue
                prefix = "  " * depth
                if item.is_dir():
                    entries.append(f"{prefix}{item.name}/")
                    if recursive:
                        _list(item, depth + 1)
                else:
                    size = item.stat().st_size
                    entries.append(f"{prefix}{item.name} ({_human_size(size)})")

        _list(p)
        output = f"Directory: {p}\n" + "\n".join(entries[:500])
        if len(entries) > 500:
            output += f"\n... and {len(entries) - 500} more entries"
        return ToolResult(tool_call_id="", output=output, success=True)


class SearchFilesTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(ToolMetadata(
            name="search_files",
            description="Search for files matching a pattern or containing text (grep-like).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to search in"},
                    "pattern": {"type": "string", "description": "Text pattern to search for (regex supported)"},
                    "file_pattern": {"type": "string", "description": "Glob pattern for filenames (e.g., '*.py')"},
                    "max_results": {"type": "integer", "description": "Max results (default: 50)"},
                },
                "required": ["path", "pattern"],
            },
            permissions=[Permission.READ],
        ))

    async def _run(self, path: str, pattern: str, file_pattern: str = "*", max_results: int = 50, **kwargs: Any) -> ToolResult:
        import re
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return ToolResult(tool_call_id="", output=f"Path not found: {path}", success=False)

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(tool_call_id="", output=f"Invalid regex: {e}", success=False)

        results = []
        ignore = {".git", "node_modules", "__pycache__", ".venv", "venv"}

        for fp in p.rglob(file_pattern):
            if any(part in ignore for part in fp.parts):
                continue
            if not fp.is_file():
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{fp}:{i}: {line.strip()}")
                        if len(results) >= max_results:
                            break
            except (PermissionError, OSError):
                continue
            if len(results) >= max_results:
                break

        output = "\n".join(results) if results else "No matches found."
        return ToolResult(tool_call_id="", output=output, success=True)


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"
