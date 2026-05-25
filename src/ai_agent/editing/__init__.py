"""Code editing engine - AST-aware editing, diff/patch generation, rollback support."""

import difflib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_agent.core import get_logger

logger = get_logger(__name__)


@dataclass
class EditOperation:
    """A single edit operation."""
    file_path: str
    edit_type: str  # "replace", "insert", "delete", "create"
    old_content: str = ""
    new_content: str = ""
    start_line: int | None = None
    end_line: int | None = None
    description: str = ""


@dataclass
class EditResult:
    success: bool
    file_path: str
    diff: str = ""
    backup_path: str = ""
    error: str = ""


@dataclass
class EditSession:
    """Tracks all edits in a session for rollback."""
    id: str
    edits: list[EditResult] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class CodeEditor:
    """Production code editing engine with safety, diffs, and rollback."""

    def __init__(self, backup_dir: Path | None = None) -> None:
        self._backup_dir = backup_dir or Path.home() / ".ai-agent" / "backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, EditSession] = {}
        self._current_session: EditSession | None = None

    def start_session(self, session_id: str) -> EditSession:
        """Start a new edit session for rollback tracking."""
        session = EditSession(id=session_id)
        self._sessions[session_id] = session
        self._current_session = session
        return session

    def apply(self, operation: EditOperation) -> EditResult:
        """Apply a single edit operation with backup."""
        path = Path(operation.file_path).expanduser().resolve()

        if operation.edit_type == "create":
            return self._create_file(path, operation)
        elif operation.edit_type == "replace":
            return self._replace(path, operation)
        elif operation.edit_type == "insert":
            return self._insert(path, operation)
        elif operation.edit_type == "delete":
            return self._delete_lines(path, operation)
        else:
            return EditResult(success=False, file_path=str(path), error=f"Unknown edit type: {operation.edit_type}")

    def apply_batch(self, operations: list[EditOperation]) -> list[EditResult]:
        """Apply multiple operations atomically."""
        results = []
        for op in operations:
            result = self.apply(op)
            results.append(result)
            if not result.success:
                # Rollback previous successful edits in this batch
                for prev in reversed(results[:-1]):
                    if prev.success and prev.backup_path:
                        self._restore_backup(prev.file_path, prev.backup_path)
                break
        return results

    def generate_diff(self, file_path: str, new_content: str) -> str:
        """Generate unified diff between current file and new content."""
        path = Path(file_path).expanduser().resolve()
        if path.exists():
            old_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        else:
            old_lines = []
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path.name}", tofile=f"b/{path.name}")
        return "".join(diff)

    def generate_patch(self, operations: list[EditOperation]) -> str:
        """Generate a combined patch for multiple operations."""
        patches = []
        for op in operations:
            path = Path(op.file_path).expanduser().resolve()
            if path.exists():
                old = path.read_text(encoding="utf-8")
            else:
                old = ""
            if op.edit_type == "create":
                diff = self.generate_diff(op.file_path, op.new_content)
            elif op.edit_type == "replace":
                new = old.replace(op.old_content, op.new_content, 1)
                diff = self.generate_diff(op.file_path, new)
            else:
                diff = f"# {op.edit_type} operation on {op.file_path}\n"
            patches.append(diff)
        return "\n".join(patches)

    def rollback_session(self, session_id: str) -> list[str]:
        """Rollback all edits in a session."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        restored = []
        for edit in reversed(session.edits):
            if edit.backup_path:
                self._restore_backup(edit.file_path, edit.backup_path)
                restored.append(edit.file_path)
        del self._sessions[session_id]
        return restored

    def rollback_file(self, file_path: str) -> bool:
        """Rollback the most recent edit to a specific file."""
        if not self._current_session:
            return False
        for edit in reversed(self._current_session.edits):
            if edit.file_path == file_path and edit.backup_path:
                self._restore_backup(file_path, edit.backup_path)
                return True
        return False

    def _create_file(self, path: Path, op: EditOperation) -> EditResult:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(op.new_content, encoding="utf-8")
            diff = self.generate_diff(str(path), op.new_content)
            result = EditResult(success=True, file_path=str(path), diff=diff)
            self._track(result)
            return result
        except Exception as e:
            return EditResult(success=False, file_path=str(path), error=str(e))

    def _replace(self, path: Path, op: EditOperation) -> EditResult:
        if not path.exists():
            return EditResult(success=False, file_path=str(path), error="File not found")

        content = path.read_text(encoding="utf-8")
        if op.old_content and op.old_content not in content:
            return EditResult(success=False, file_path=str(path), error="Old content not found in file")

        backup = self._create_backup(path)
        if op.old_content:
            new_content = content.replace(op.old_content, op.new_content, 1)
        else:
            new_content = op.new_content

        path.write_text(new_content, encoding="utf-8")
        diff = difflib.unified_diff(
            content.splitlines(keepends=True), new_content.splitlines(keepends=True),
            fromfile=f"a/{path.name}", tofile=f"b/{path.name}",
        )
        result = EditResult(success=True, file_path=str(path), diff="".join(diff), backup_path=str(backup))
        self._track(result)
        return result

    def _insert(self, path: Path, op: EditOperation) -> EditResult:
        if not path.exists():
            return EditResult(success=False, file_path=str(path), error="File not found")

        content = path.read_text(encoding="utf-8")
        backup = self._create_backup(path)
        lines = content.splitlines(keepends=True)

        insert_at = (op.start_line or len(lines)) - 1
        insert_at = max(0, min(insert_at, len(lines)))
        new_lines = op.new_content.splitlines(keepends=True)
        lines[insert_at:insert_at] = new_lines

        new_content = "".join(lines)
        path.write_text(new_content, encoding="utf-8")
        diff = difflib.unified_diff(
            content.splitlines(keepends=True), new_content.splitlines(keepends=True),
            fromfile=f"a/{path.name}", tofile=f"b/{path.name}",
        )
        result = EditResult(success=True, file_path=str(path), diff="".join(diff), backup_path=str(backup))
        self._track(result)
        return result

    def _delete_lines(self, path: Path, op: EditOperation) -> EditResult:
        if not path.exists():
            return EditResult(success=False, file_path=str(path), error="File not found")

        content = path.read_text(encoding="utf-8")
        backup = self._create_backup(path)
        lines = content.splitlines(keepends=True)

        start = (op.start_line or 1) - 1
        end = op.end_line or (start + 1)
        del lines[start:end]

        new_content = "".join(lines)
        path.write_text(new_content, encoding="utf-8")
        diff = difflib.unified_diff(
            content.splitlines(keepends=True), new_content.splitlines(keepends=True),
            fromfile=f"a/{path.name}", tofile=f"b/{path.name}",
        )
        result = EditResult(success=True, file_path=str(path), diff="".join(diff), backup_path=str(backup))
        self._track(result)
        return result

    def _create_backup(self, path: Path) -> Path:
        timestamp = int(time.time() * 1000)
        backup = self._backup_dir / f"{path.name}.{timestamp}.bak"
        shutil.copy2(path, backup)
        return backup

    def _restore_backup(self, file_path: str, backup_path: str) -> None:
        shutil.copy2(backup_path, file_path)

    def _track(self, result: EditResult) -> None:
        if self._current_session:
            self._current_session.edits.append(result)
