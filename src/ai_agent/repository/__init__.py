"""Repository understanding engine - scans, analyzes, and maps codebases."""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_agent.core import get_logger

logger = get_logger(__name__)

FRAMEWORK_INDICATORS: dict[str, dict[str, list[str]]] = {
    "python": {
        "django": ["manage.py", "settings.py", "urls.py", "wsgi.py"],
        "fastapi": ["main.py:FastAPI", "requirements.txt:fastapi"],
        "flask": ["app.py:Flask", "requirements.txt:flask"],
        "pytorch": ["requirements.txt:torch", "model.py", "train.py"],
    },
    "javascript": {
        "react": ["package.json:react", "src/App.jsx", "src/App.tsx"],
        "nextjs": ["next.config.js", "pages/", "app/"],
        "express": ["package.json:express", "server.js", "app.js"],
        "vue": ["package.json:vue", "src/App.vue"],
    },
    "rust": {"actix": ["Cargo.toml:actix-web"], "tokio": ["Cargo.toml:tokio"]},
    "go": {"gin": ["go.mod:gin-gonic"], "fiber": ["go.mod:gofiber"]},
}

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".tox", "target", ".next"}


@dataclass
class FileInfo:
    path: str
    language: str
    size: int
    lines: int
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)


@dataclass
class RepoAnalysis:
    root: str
    languages: dict[str, int] = field(default_factory=dict)
    frameworks: list[str] = field(default_factory=list)
    architecture: str = ""
    entrypoints: list[str] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    file_count: int = 0
    total_lines: int = 0
    modules: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""


LANG_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".jsx": "javascript", ".rs": "rust", ".go": "go", ".java": "java",
    ".rb": "ruby", ".php": "php", ".c": "c", ".cpp": "cpp", ".h": "c",
    ".cs": "csharp", ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".vue": "vue", ".svelte": "svelte", ".md": "markdown", ".yml": "yaml",
    ".yaml": "yaml", ".json": "json", ".toml": "toml", ".sql": "sql",
}


class RepositoryAnalyzer:
    """Analyzes repository structure, frameworks, and architecture."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    def analyze(self) -> RepoAnalysis:
        """Full repository analysis."""
        analysis = RepoAnalysis(root=str(self._root))

        files = self._scan_files()
        analysis.file_count = len(files)
        analysis.total_lines = sum(f.lines for f in files)

        # Language distribution
        lang_lines: Counter[str] = Counter()
        for f in files:
            lang_lines[f.language] += f.lines
        analysis.languages = dict(lang_lines.most_common(10))

        # Detect frameworks
        analysis.frameworks = self._detect_frameworks(files)

        # Find entrypoints
        analysis.entrypoints = self._find_entrypoints(files)

        # Parse dependencies
        analysis.dependencies = self._parse_dependencies()

        # Detect architecture
        analysis.architecture = self._detect_architecture(files)

        # Map modules
        analysis.modules = self._map_modules(files)

        # Generate summary
        primary_lang = lang_lines.most_common(1)[0][0] if lang_lines else "unknown"
        analysis.summary = (
            f"{primary_lang.title()} project with {analysis.file_count} files, "
            f"{analysis.total_lines} lines. "
            f"Frameworks: {', '.join(analysis.frameworks) or 'none detected'}. "
            f"Architecture: {analysis.architecture}."
        )

        return analysis

    def _scan_files(self) -> list[FileInfo]:
        """Scan all source files."""
        files = []
        for fp in self._root.rglob("*"):
            if any(p in IGNORE_DIRS for p in fp.parts):
                continue
            if not fp.is_file():
                continue
            lang = LANG_MAP.get(fp.suffix, "")
            if not lang:
                continue
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
                lines = content.count("\n") + 1
                imports = self._extract_imports(content, lang)
                symbols = self._extract_symbols(content, lang)
                files.append(FileInfo(
                    path=str(fp.relative_to(self._root)),
                    language=lang, size=fp.stat().st_size, lines=lines,
                    imports=imports, symbols=symbols,
                ))
            except (PermissionError, OSError):
                continue
        return files

    def _detect_frameworks(self, files: list[FileInfo]) -> list[str]:
        """Detect frameworks from file patterns and imports."""
        detected = []
        file_paths = {f.path for f in files}
        all_content_cache: dict[str, str] = {}

        for lang, frameworks in FRAMEWORK_INDICATORS.items():
            for framework, indicators in frameworks.items():
                for indicator in indicators:
                    if ":" in indicator:
                        file_part, content_part = indicator.split(":", 1)
                        for fp in file_paths:
                            if fp.endswith(file_part):
                                if fp not in all_content_cache:
                                    try:
                                        all_content_cache[fp] = (self._root / fp).read_text(errors="ignore")
                                    except OSError:
                                        continue
                                if content_part in all_content_cache.get(fp, ""):
                                    detected.append(framework)
                                    break
                    elif indicator.endswith("/"):
                        if (self._root / indicator.rstrip("/")).is_dir():
                            detected.append(framework)
                    elif any(fp.endswith(indicator) for fp in file_paths):
                        detected.append(framework)

        return list(set(detected))

    def _find_entrypoints(self, files: list[FileInfo]) -> list[str]:
        """Find likely entrypoints."""
        entrypoint_patterns = [
            "main.py", "app.py", "server.py", "index.ts", "index.js",
            "main.go", "main.rs", "Main.java", "manage.py", "cli.py",
        ]
        found = []
        for f in files:
            basename = Path(f.path).name
            if basename in entrypoint_patterns:
                found.append(f.path)
            # Check for if __name__ == "__main__" pattern
            if f.language == "python" and "__main__" in " ".join(f.symbols):
                found.append(f.path)
        return found[:10]

    def _parse_dependencies(self) -> dict[str, list[str]]:
        """Parse dependency files."""
        deps: dict[str, list[str]] = {}

        # Python
        for dep_file in ["requirements.txt", "pyproject.toml", "setup.py"]:
            path = self._root / dep_file
            if path.exists():
                content = path.read_text(errors="ignore")
                if dep_file == "requirements.txt":
                    deps["python"] = [l.split("==")[0].split(">=")[0].strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
                elif dep_file == "pyproject.toml":
                    # Simple extraction from dependencies section
                    in_deps = False
                    for line in content.splitlines():
                        if "dependencies" in line and "=" in line:
                            in_deps = True
                            continue
                        if in_deps:
                            if line.strip().startswith("]"):
                                in_deps = False
                            elif '"' in line:
                                pkg = line.strip().strip('",').split(">=")[0].split("==")[0].split("[")[0]
                                if pkg:
                                    deps.setdefault("python", []).append(pkg)

        # Node.js
        pkg_json = self._root / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text())
                deps["node"] = list(data.get("dependencies", {}).keys())
            except json.JSONDecodeError:
                pass

        # Rust
        cargo = self._root / "Cargo.toml"
        if cargo.exists():
            content = cargo.read_text(errors="ignore")
            in_deps = False
            for line in content.splitlines():
                if "[dependencies]" in line:
                    in_deps = True
                    continue
                if in_deps and line.startswith("["):
                    break
                if in_deps and "=" in line:
                    deps.setdefault("rust", []).append(line.split("=")[0].strip())

        return deps

    def _detect_architecture(self, files: list[FileInfo]) -> str:
        """Detect project architecture pattern."""
        paths = [f.path for f in files]
        path_str = "\n".join(paths)

        if any("src/" in p and "/controllers/" in p for p in paths):
            return "MVC"
        if any("/domain/" in p or "/entities/" in p for p in paths):
            return "Clean Architecture / DDD"
        if any("/services/" in p and "/repositories/" in p for p in paths):
            return "Layered (Service/Repository)"
        if any("/api/" in p and "/core/" in p for p in paths):
            return "Hexagonal / Ports & Adapters"
        if any("/handlers/" in p or "/routes/" in p for p in paths):
            return "Handler-based"
        if any("/components/" in p for p in paths):
            return "Component-based"
        if any("/cmd/" in p and "/pkg/" in p for p in paths):
            return "Go Standard Layout"
        return "Flat / Custom"

    def _map_modules(self, files: list[FileInfo]) -> list[dict[str, Any]]:
        """Map top-level modules/packages."""
        modules: dict[str, dict[str, Any]] = {}
        for f in files:
            parts = Path(f.path).parts
            if len(parts) > 1:
                module = parts[0] if parts[0] != "src" else (parts[1] if len(parts) > 2 else parts[0])
            else:
                module = "root"
            if module not in modules:
                modules[module] = {"name": module, "files": 0, "lines": 0, "languages": set()}
            modules[module]["files"] += 1
            modules[module]["lines"] += f.lines
            modules[module]["languages"].add(f.language)

        return [
            {"name": m["name"], "files": m["files"], "lines": m["lines"], "languages": list(m["languages"])}
            for m in sorted(modules.values(), key=lambda x: x["lines"], reverse=True)[:20]
        ]

    @staticmethod
    def _extract_imports(content: str, language: str) -> list[str]:
        """Extract import statements."""
        imports = []
        if language == "python":
            for m in re.finditer(r"^(?:from|import)\s+(\S+)", content, re.MULTILINE):
                imports.append(m.group(1))
        elif language in ("javascript", "typescript"):
            for m in re.finditer(r"(?:import|require)\s*\(?['\"]([^'\"]+)['\"]", content):
                imports.append(m.group(1))
        elif language == "go":
            for m in re.finditer(r'"([^"]+)"', content):
                if "/" in m.group(1):
                    imports.append(m.group(1))
        elif language == "rust":
            for m in re.finditer(r"use\s+([\w:]+)", content):
                imports.append(m.group(1))
        return imports[:50]

    @staticmethod
    def _extract_symbols(content: str, language: str) -> list[str]:
        """Extract top-level symbol definitions."""
        symbols = []
        if language == "python":
            for m in re.finditer(r"^(?:def|class|async def)\s+(\w+)", content, re.MULTILINE):
                symbols.append(m.group(1))
        elif language in ("javascript", "typescript"):
            for m in re.finditer(r"(?:export\s+)?(?:function|class|const|let|var)\s+(\w+)", content):
                symbols.append(m.group(1))
        elif language == "go":
            for m in re.finditer(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", content, re.MULTILINE):
                symbols.append(m.group(1))
        elif language == "rust":
            for m in re.finditer(r"^(?:pub\s+)?(?:fn|struct|enum|trait|impl)\s+(\w+)", content, re.MULTILINE):
                symbols.append(m.group(1))
        return symbols[:100]
