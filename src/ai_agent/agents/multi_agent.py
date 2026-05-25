"""Multi-agent system - specialized agents with delegation and collaboration."""

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_agent.agents.engine import AgentEngine
from ai_agent.core import Event, Message, Role, TaskStatus, get_event_bus, get_logger, get_settings
from ai_agent.memory import MemoryManager, MemoryType
from ai_agent.models import LLMClient
from ai_agent.tools import ToolRegistry, create_default_registry

logger = get_logger(__name__)


class AgentRole(str, Enum):
    MANAGER = "manager"
    CODER = "coder"
    REVIEWER = "reviewer"
    DEBUGGER = "debugger"
    RESEARCHER = "researcher"
    PLANNER = "planner"


ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.MANAGER: """You are a Manager Agent. Your job is to:
- Decompose complex tasks into subtasks
- Delegate subtasks to specialized agents
- Coordinate work between agents
- Synthesize results into a final deliverable
- Ensure quality and completeness

Respond with a JSON plan: {"subtasks": [{"agent": "coder|reviewer|debugger|researcher", "task": "description", "priority": 1-5}]}""",

    AgentRole.CODER: """You are a Coder Agent. You are an expert programmer who:
- Writes clean, production-quality code
- Follows best practices and design patterns
- Implements features completely with error handling
- Writes tests alongside implementation
- Uses tools to read existing code before making changes""",

    AgentRole.REVIEWER: """You are a Code Reviewer Agent. You:
- Review code for bugs, security issues, and best practices
- Check for edge cases and error handling
- Verify code style and consistency
- Suggest improvements with specific code examples
- Rate code quality on a 1-10 scale

Respond with: {"score": N, "issues": [...], "suggestions": [...], "approved": bool}""",

    AgentRole.DEBUGGER: """You are a Debugger Agent. You:
- Analyze error messages and stack traces
- Reproduce bugs systematically
- Identify root causes
- Propose and implement fixes
- Verify fixes work correctly""",

    AgentRole.RESEARCHER: """You are a Research Agent. You:
- Investigate codebases to understand architecture
- Find relevant code patterns and examples
- Research best practices for specific problems
- Summarize findings concisely
- Provide context for other agents""",

    AgentRole.PLANNER: """You are a Planning Agent. You:
- Break down complex projects into phases
- Define clear milestones and deliverables
- Identify dependencies between tasks
- Estimate complexity and effort
- Create actionable implementation plans

Respond with: {"phases": [{"name": str, "tasks": [...], "dependencies": [...]}]}""",
}


@dataclass
class AgentMessage:
    """Message between agents."""
    from_agent: str
    to_agent: str
    content: str
    message_type: str = "task"  # task, result, feedback, query
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubTask:
    id: str
    agent_role: AgentRole
    task: str
    priority: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    dependencies: list[str] = field(default_factory=list)


@dataclass
class MultiAgentResult:
    task: str
    subtasks: list[SubTask]
    final_result: str
    success: bool
    messages: list[AgentMessage] = field(default_factory=list)


class SpecializedAgent:
    """A specialized agent with a specific role."""

    def __init__(self, role: AgentRole, llm: LLMClient | None = None, tools: ToolRegistry | None = None) -> None:
        self.role = role
        self.id = f"{role.value}_{uuid.uuid4().hex[:6]}"
        self._llm = llm or LLMClient()
        self._tools = tools or create_default_registry()
        self._engine = AgentEngine(
            llm=self._llm,
            tools=self._tools.as_dict,
            system_prompt=ROLE_PROMPTS[role],
        )

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        """Execute a task according to this agent's role."""
        state = await self._engine.run(task, context=context, plan_first=(self.role != AgentRole.REVIEWER))
        if state.messages:
            # Return the last assistant message
            for msg in reversed(state.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    return msg["content"]
        return f"Agent {self.id} completed task but produced no output."


class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents for complex tasks."""

    def __init__(self, llm: LLMClient | None = None, memory: MemoryManager | None = None) -> None:
        self._llm = llm or LLMClient()
        self._memory = memory or MemoryManager()
        self._event_bus = get_event_bus()
        self._agents: dict[AgentRole, SpecializedAgent] = {}
        self._message_log: list[AgentMessage] = []

    def _get_agent(self, role: AgentRole) -> SpecializedAgent:
        if role not in self._agents:
            self._agents[role] = SpecializedAgent(role=role, llm=self._llm)
        return self._agents[role]

    async def execute(self, task: str, strategy: str = "sequential") -> MultiAgentResult:
        """Execute a complex task using multiple agents."""
        await self._emit("orchestration_start", {"task": task, "strategy": strategy})

        # Phase 1: Planning
        planner = self._get_agent(AgentRole.PLANNER)
        plan_result = await planner.execute(f"Create an implementation plan for: {task}")

        # Phase 2: Parse subtasks from plan
        subtasks = self._parse_subtasks(plan_result, task)

        if not subtasks:
            # Fallback: single coder agent
            subtasks = [SubTask(id="main", agent_role=AgentRole.CODER, task=task, priority=1)]

        # Phase 3: Execute subtasks
        if strategy == "parallel":
            results = await self._execute_parallel(subtasks)
        else:
            results = await self._execute_sequential(subtasks)

        # Phase 4: Review (if coder was involved)
        coder_results = [st for st in results if st.agent_role == AgentRole.CODER and st.result]
        if coder_results:
            reviewer = self._get_agent(AgentRole.REVIEWER)
            review_context = "\n\n".join(f"Code from {st.id}:\n{st.result}" for st in coder_results)
            review = await reviewer.execute(f"Review this code:\n{review_context}")
            self._message_log.append(AgentMessage(
                from_agent=reviewer.id, to_agent="orchestrator",
                content=review, message_type="feedback",
            ))

        # Phase 5: Synthesize final result
        manager = self._get_agent(AgentRole.MANAGER)
        synthesis_input = "\n\n".join(f"[{st.agent_role.value}] {st.task}:\n{st.result}" for st in results if st.result)
        final = await manager.execute(f"Synthesize these results into a final deliverable for: {task}\n\nResults:\n{synthesis_input}")

        # Store in memory
        self._memory.store_episode(
            task=task,
            actions=[st.task for st in results],
            outcome=final[:500],
            success=all(st.status == TaskStatus.COMPLETED for st in results),
        )

        await self._emit("orchestration_complete", {"task": task, "subtask_count": len(results)})

        return MultiAgentResult(
            task=task, subtasks=results, final_result=final,
            success=all(st.status == TaskStatus.COMPLETED for st in results),
            messages=self._message_log.copy(),
        )

    async def _execute_sequential(self, subtasks: list[SubTask]) -> list[SubTask]:
        """Execute subtasks one by one, passing context forward."""
        context: dict[str, Any] = {}
        for st in sorted(subtasks, key=lambda x: x.priority):
            agent = self._get_agent(st.agent_role)
            st.status = TaskStatus.RUNNING
            await self._emit("subtask_start", {"id": st.id, "agent": st.agent_role.value})

            try:
                result = await agent.execute(st.task, context=context)
                st.result = result
                st.status = TaskStatus.COMPLETED
                context[st.id] = result
            except Exception as e:
                st.result = f"Error: {e}"
                st.status = TaskStatus.FAILED

            self._message_log.append(AgentMessage(
                from_agent=agent.id, to_agent="orchestrator",
                content=st.result, message_type="result",
            ))
        return subtasks

    async def _execute_parallel(self, subtasks: list[SubTask]) -> list[SubTask]:
        """Execute independent subtasks in parallel."""
        # Group by dependency
        independent = [st for st in subtasks if not st.dependencies]
        dependent = [st for st in subtasks if st.dependencies]

        # Run independent tasks in parallel
        async def _run(st: SubTask) -> SubTask:
            agent = self._get_agent(st.agent_role)
            st.status = TaskStatus.RUNNING
            try:
                st.result = await agent.execute(st.task)
                st.status = TaskStatus.COMPLETED
            except Exception as e:
                st.result = f"Error: {e}"
                st.status = TaskStatus.FAILED
            return st

        completed = await asyncio.gather(*[_run(st) for st in independent])

        # Run dependent tasks sequentially
        context = {st.id: st.result for st in completed if st.status == TaskStatus.COMPLETED}
        for st in dependent:
            agent = self._get_agent(st.agent_role)
            st.status = TaskStatus.RUNNING
            try:
                st.result = await agent.execute(st.task, context=context)
                st.status = TaskStatus.COMPLETED
                context[st.id] = st.result
            except Exception as e:
                st.result = f"Error: {e}"
                st.status = TaskStatus.FAILED

        return list(completed) + dependent

    def _parse_subtasks(self, plan_result: str, original_task: str) -> list[SubTask]:
        """Parse planner output into subtasks."""
        import json
        try:
            # Try to extract JSON from the plan
            start = plan_result.find("{")
            end = plan_result.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(plan_result[start:end])
                subtasks = []
                for phase in data.get("phases", data.get("subtasks", [])):
                    if isinstance(phase, dict):
                        tasks = phase.get("tasks", [phase])
                        for t in tasks:
                            if isinstance(t, str):
                                subtasks.append(SubTask(
                                    id=f"task_{len(subtasks)}",
                                    agent_role=AgentRole.CODER,
                                    task=t,
                                    priority=len(subtasks) + 1,
                                ))
                            elif isinstance(t, dict):
                                role_str = t.get("agent", "coder")
                                role = AgentRole(role_str) if role_str in AgentRole._value2member_map_ else AgentRole.CODER
                                subtasks.append(SubTask(
                                    id=f"task_{len(subtasks)}",
                                    agent_role=role,
                                    task=t.get("task", t.get("description", "")),
                                    priority=t.get("priority", len(subtasks) + 1),
                                    dependencies=t.get("dependencies", []),
                                ))
                return subtasks
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return []

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        await self._event_bus.publish(Event(type=f"multi_agent.{event_type}", data=data))
