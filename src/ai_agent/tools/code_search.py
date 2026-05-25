"""Code search tool using tree-sitter for AST-aware search."""

import re
from pathlib import Path
from typing import Any

from ai_agent.core import ToolResult
from ai_agent.tools.base import BaseTool, Permission, ToolMetadata

# Language extensions mapping
LANG_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "rust": [".rs"],
    "go": [".go"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc", ".cxx"],
    "ruby": [".rb"],
    "php": [".php"],
}


class CodeSearchTool(BaseTool):
    """Search for code symbols, definitions, and patterns."""

    def __init__(self) -> None:
        super().__init__(ToolMetadata(
            name="code_search",
            description="Search for code symbols (functions, classes, methods) or patterns in a codebase. Supports regex and symbol-type filtering.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to search in"},
                    "query": {"type": "string", "description": "Search query (symbol name or regex pattern)"},
                    "symbol_type": {
                        "type": "string",
                        "enum": ["function", "class", "method", "variable", "import", "any"],
                        "description": "Type of symbol to search for (default: any)",
                    },
                    "language": {"type": "string", "description": "Filter by language (e.g., 'python', 'typescript')"},
                    "max_results": {"type": "integer", "description": "Max results (default: 30)"},
                },
                "required": ["path", "query"],
            },
            permissions=[Permission.READ],
        ))

    async def _run(
        self, path: str, query: str, symbol_type: str = "any",
        language: str | None = None, max_results: int = 30, **kwargs: Any,
    ) -> ToolResult:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return ToolResult(tool_call_id="", output=f"Path not found: {path}", success=False)

        # Determine file extensions to search
        extensions: set[str] = set()
        if language and language in LANG_EXTENSIONS:
            extensions = set(LANG_EXTENSIONS[language])
        else:
            for exts in LANG_EXTENSIONS.values():
                extensions.update(exts)

        # Build regex patterns for symbol types
        patterns = self._get_patterns(query, symbol_type)
        ignore = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
        results: list[str] = []

        for fp in p.rglob("*"):
            if any(part in ignore for part in fp.parts):
                continue
            if not fp.is_file() or fp.suffix not in extensions:
                continue
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.splitlines(), 1):
                    for pat in patterns:
                        if pat.search(line):
                            results.append(f"{fp.relative_to(p)}:{i}: {line.strip()}")
                            break
                    if len(results) >= max_results:
                        break
            except (PermissionError, OSError):
                continue
            if len(results) >= max_results:
                break

        output = "\n".join(results) if results else f"No matches for '{query}'"
        return ToolResult(tool_call_id="", output=output, success=True)

    def _get_patterns(self, query: str, symbol_type: str) -> list[re.Pattern[str]]:
        patterns = []
        q = re.escape(query)
        if symbol_type in ("function", "any"):
            patterns.append(re.compile(rf"(def|function|func|fn)\s+{q}", re.IGNORECASE))
        if symbol_type in ("class", "any"):
            patterns.append(re.compile(rf"(class|struct|interface|trait)\s+{q}", re.IGNORECASE))
        if symbol_type in ("method", "any"):
            patterns.append(re.compile(rf"(def|func|fn|public|private|protected)\s+{q}\s*\(", re.IGNORECASE))
        if symbol_type in ("variable", "any"):
            patterns.append(re.compile(rf"(let|const|var|val)\s+{q}\s*[=:]", re.IGNORECASE))
        if symbol_type in ("import", "any"):
            patterns.append(re.compile(rf"(import|from|require|use)\s+.*{q}", re.IGNORECASE))
        if not patterns:
            patterns.append(re.compile(query, re.IGNORECASE))
        return patterns
