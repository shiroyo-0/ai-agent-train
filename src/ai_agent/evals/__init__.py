"""Self-improvement system - reflection, scoring, prompt optimization, experience learning."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_agent.core import Message, Role, get_logger, get_settings
from ai_agent.memory import MemoryManager, MemoryType
from ai_agent.models import LLMClient

logger = get_logger(__name__)


@dataclass
class ExecutionRecord:
    task: str
    result: str
    success: bool
    score: float
    duration: float
    tools_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    feedback: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class PromptVariant:
    prompt: str
    score: float = 0.0
    uses: int = 0
    successes: int = 0


class SelfImprover:
    """Agent self-improvement through reflection, scoring, and optimization."""

    def __init__(self, llm: LLMClient | None = None, memory: MemoryManager | None = None) -> None:
        self._llm = llm or LLMClient()
        self._memory = memory or MemoryManager()
        self._settings = get_settings()
        self._records: list[ExecutionRecord] = []
        self._prompt_variants: dict[str, list[PromptVariant]] = {}
        self._data_dir = self._settings.data_dir / "self_improvement"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

    async def reflect(self, task: str, result: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Deep self-reflection on task execution."""
        messages = [
            Message(role=Role.SYSTEM, content=(
                "You are a self-reflection engine. Analyze the task execution and provide:\n"
                "1. What went well\n2. What could be improved\n3. Specific lessons learned\n4. Score (1-10)\n"
                "Return JSON: {\"score\": N, \"went_well\": [...], \"improvements\": [...], \"lessons\": [...], \"meta_insights\": str}"
            )),
            Message(role=Role.USER, content=f"Task: {task}\n\nResult: {result}\n\nContext: {json.dumps(context or {})}"),
        ]
        response = await self._llm.complete(messages)
        try:
            reflection = json.loads(response.message.content.strip().strip("```json").strip("```"))
        except (json.JSONDecodeError, ValueError):
            reflection = {"score": 5, "went_well": [], "improvements": ["Unable to parse reflection"], "lessons": [], "meta_insights": ""}

        # Store lessons in long-term memory
        for lesson in reflection.get("lessons", []):
            self._memory.store(lesson, MemoryType.LONG_TERM, tags=["lesson", "self_improvement"])

        return reflection

    async def score_output(self, task: str, output: str, criteria: list[str] | None = None) -> dict[str, Any]:
        """Score an output against quality criteria."""
        default_criteria = ["correctness", "completeness", "code_quality", "efficiency", "clarity"]
        criteria = criteria or default_criteria

        messages = [
            Message(role=Role.SYSTEM, content=(
                f"Score this output on each criterion (1-10). Criteria: {criteria}\n"
                "Return JSON: {\"scores\": {\"criterion\": N, ...}, \"overall\": N, \"reasoning\": str}"
            )),
            Message(role=Role.USER, content=f"Task: {task}\nOutput: {output}"),
        ]
        response = await self._llm.complete(messages)
        try:
            return json.loads(response.message.content.strip().strip("```json").strip("```"))
        except (json.JSONDecodeError, ValueError):
            return {"scores": {c: 5 for c in criteria}, "overall": 5, "reasoning": "Unable to score"}

    async def optimize_prompt(self, base_prompt: str, task_type: str, examples: list[dict[str, Any]] | None = None) -> str:
        """Optimize a prompt based on past performance."""
        # Get relevant past experiences
        past = self._memory.recall(f"prompt optimization {task_type}", limit=5)
        past_context = "\n".join(p.content for p in past) if past else "No prior optimization data."

        messages = [
            Message(role=Role.SYSTEM, content=(
                "You are a prompt optimization engine. Given a base prompt and performance data, "
                "generate an improved version that will produce better results. "
                "Return ONLY the improved prompt text."
            )),
            Message(role=Role.USER, content=(
                f"Base prompt:\n{base_prompt}\n\n"
                f"Task type: {task_type}\n"
                f"Past performance context:\n{past_context}\n\n"
                f"Examples of good/bad outputs:\n{json.dumps(examples or [])}\n\n"
                "Generate an improved prompt:"
            )),
        ]
        response = await self._llm.complete(messages)
        optimized = response.message.content.strip()

        # Track the variant
        if task_type not in self._prompt_variants:
            self._prompt_variants[task_type] = []
        self._prompt_variants[task_type].append(PromptVariant(prompt=optimized))

        # Store optimization in memory
        self._memory.store(
            f"Optimized prompt for {task_type}: {optimized[:200]}",
            MemoryType.LONG_TERM,
            tags=["prompt_optimization", task_type],
        )

        return optimized

    async def learn_from_failure(self, task: str, error: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Analyze a failure and extract actionable improvements."""
        messages = [
            Message(role=Role.SYSTEM, content=(
                "Analyze this failure and provide:\n"
                "1. Root cause analysis\n2. Prevention strategy\n3. Recovery steps\n"
                "Return JSON: {\"root_cause\": str, \"prevention\": str, \"recovery_steps\": [...], \"pattern\": str}"
            )),
            Message(role=Role.USER, content=f"Task: {task}\nError: {error}\nContext: {json.dumps(context or {})}"),
        ]
        response = await self._llm.complete(messages)
        try:
            analysis = json.loads(response.message.content.strip().strip("```json").strip("```"))
        except (json.JSONDecodeError, ValueError):
            analysis = {"root_cause": "Unknown", "prevention": "", "recovery_steps": [], "pattern": ""}

        # Store failure pattern for future avoidance
        self._memory.store(
            f"Failure pattern: {analysis.get('pattern', '')} - Prevention: {analysis.get('prevention', '')}",
            MemoryType.LONG_TERM,
            metadata={"task": task, "error": error},
            tags=["failure_pattern", "prevention"],
        )

        return analysis

    async def suggest_workflow_improvements(self) -> list[dict[str, Any]]:
        """Analyze execution history and suggest workflow improvements."""
        if not self._records:
            return []

        # Analyze patterns
        success_rate = sum(1 for r in self._records if r.success) / len(self._records)
        avg_duration = sum(r.duration for r in self._records) / len(self._records)
        common_errors = {}
        for r in self._records:
            for e in r.errors:
                common_errors[e] = common_errors.get(e, 0) + 1

        messages = [
            Message(role=Role.SYSTEM, content="Analyze agent performance data and suggest improvements. Return JSON array of suggestions."),
            Message(role=Role.USER, content=(
                f"Stats: success_rate={success_rate:.2f}, avg_duration={avg_duration:.1f}s, "
                f"total_tasks={len(self._records)}\n"
                f"Common errors: {json.dumps(dict(sorted(common_errors.items(), key=lambda x: -x[1])[:5]))}\n"
                f"Recent tasks: {json.dumps([{'task': r.task[:50], 'success': r.success, 'score': r.score} for r in self._records[-10:]])}"
            )),
        ]
        response = await self._llm.complete(messages)
        try:
            return json.loads(response.message.content.strip().strip("```json").strip("```"))
        except (json.JSONDecodeError, ValueError):
            return [{"suggestion": "Collect more execution data for better analysis"}]

    def record_execution(self, record: ExecutionRecord) -> None:
        """Record a task execution for learning."""
        self._records.append(record)
        self._memory.store_episode(
            task=record.task,
            actions=record.tools_used,
            outcome=record.result[:500],
            success=record.success,
        )
        self._persist_state()

    def get_performance_stats(self) -> dict[str, Any]:
        """Get aggregate performance statistics."""
        if not self._records:
            return {"total": 0}
        return {
            "total": len(self._records),
            "success_rate": sum(1 for r in self._records if r.success) / len(self._records),
            "avg_score": sum(r.score for r in self._records) / len(self._records),
            "avg_duration": sum(r.duration for r in self._records) / len(self._records),
            "total_tools_used": sum(len(r.tools_used) for r in self._records),
        }

    def _persist_state(self) -> None:
        """Save improvement state to disk."""
        data = {
            "records": [
                {"task": r.task, "success": r.success, "score": r.score, "duration": r.duration,
                 "tools_used": r.tools_used, "errors": r.errors, "timestamp": r.timestamp}
                for r in self._records[-1000:]  # Keep last 1000
            ],
        }
        (self._data_dir / "state.json").write_text(json.dumps(data))

    def _load_state(self) -> None:
        """Load persisted state."""
        path = self._data_dir / "state.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for r in data.get("records", []):
                self._records.append(ExecutionRecord(
                    task=r["task"], result="", success=r["success"],
                    score=r["score"], duration=r["duration"],
                    tools_used=r.get("tools_used", []), errors=r.get("errors", []),
                    timestamp=r.get("timestamp", 0),
                ))
        except Exception as e:
            logger.warning("failed_to_load_improvement_state", error=str(e))
