"""Agent engine - core reasoning loop with planning, execution, and reflection."""

import asyncio
import json
import uuid
from typing import Any

from ai_agent.core import (
    AgentState, Event, Message, Role, TaskStatus, ToolCall, ToolResult,
    get_event_bus, get_logger, get_settings,
)
from ai_agent.models import LLMClient

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert AI coding agent. You solve tasks step-by-step.

CAPABILITIES:
- Read, write, and edit files
- Execute shell commands
- Search code and repositories
- Plan and decompose complex tasks
- Debug and fix errors autonomously

RULES:
1. Think step-by-step before acting
2. Use tools to gather information before making changes
3. Verify your work after making changes
4. If something fails, analyze the error and try a different approach
5. Be precise and minimal in your changes

When you have completed the task, respond with your final answer without tool calls."""


class Planner:
    """Decomposes tasks into executable steps."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def create_plan(self, task: str, context: str = "") -> list[str]:
        messages = [
            Message(role=Role.SYSTEM, content="You are a task planner. Break down the task into concrete, actionable steps. Return a JSON array of strings."),
            Message(role=Role.USER, content=f"Task: {task}\n\nContext: {context}\n\nReturn ONLY a JSON array of step strings."),
        ]
        response = await self._llm.complete(messages)
        try:
            steps = json.loads(response.message.content.strip().strip("```json").strip("```"))
            return steps if isinstance(steps, list) else [task]
        except (json.JSONDecodeError, ValueError):
            return [task]

    async def replan(self, task: str, completed: list[str], failed_step: str, error: str) -> list[str]:
        messages = [
            Message(role=Role.SYSTEM, content="You are a task planner. Given a failed step, create a revised plan. Return a JSON array of strings."),
            Message(role=Role.USER, content=f"Task: {task}\nCompleted: {completed}\nFailed: {failed_step}\nError: {error}\n\nReturn ONLY a JSON array of revised remaining steps."),
        ]
        response = await self._llm.complete(messages)
        try:
            steps = json.loads(response.message.content.strip().strip("```json").strip("```"))
            return steps if isinstance(steps, list) else [task]
        except (json.JSONDecodeError, ValueError):
            return [task]


class Reflector:
    """Evaluates agent outputs and provides feedback."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def evaluate(self, task: str, result: str, context: str = "") -> dict[str, Any]:
        messages = [
            Message(role=Role.SYSTEM, content="Evaluate if the task was completed successfully. Return JSON with: {\"success\": bool, \"score\": 0-10, \"feedback\": str, \"issues\": [str]}"),
            Message(role=Role.USER, content=f"Task: {task}\nResult: {result}\nContext: {context}"),
        ]
        response = await self._llm.complete(messages)
        try:
            return json.loads(response.message.content.strip().strip("```json").strip("```"))
        except (json.JSONDecodeError, ValueError):
            return {"success": True, "score": 7, "feedback": "Unable to evaluate", "issues": []}


class AgentEngine:
    """Core agent engine with reasoning loop, tool execution, and self-correction."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._tools = tools or {}
        self._system_prompt = system_prompt or SYSTEM_PROMPT
        self._planner = Planner(self._llm)
        self._reflector = Reflector(self._llm)
        self._event_bus = get_event_bus()
        self._settings = get_settings()
        self.state: AgentState | None = None

    def register_tool(self, tool: Any) -> None:
        self._tools[tool.name] = tool

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        schemas = []
        for tool in self._tools.values():
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters if hasattr(tool, "parameters") else {"type": "object", "properties": {}},
                },
            }
            schemas.append(schema)
        return schemas

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        max_iterations: int | None = None,
        plan_first: bool = True,
    ) -> AgentState:
        """Execute a task with the full reasoning loop."""
        session_id = str(uuid.uuid4())
        self.state = AgentState(
            session_id=session_id,
            task=task,
            status=TaskStatus.RUNNING,
            context=context or {},
        )

        max_iter = max_iterations or self._settings.max_iterations
        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=self._system_prompt),
        ]

        # Planning phase
        if plan_first:
            plan = await self._planner.create_plan(task, json.dumps(context or {}))
            self.state.plan = plan
            await self._emit("plan_created", {"plan": plan})
            task_with_plan = f"{task}\n\nPlan:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
            messages.append(Message(role=Role.USER, content=task_with_plan))
        else:
            messages.append(Message(role=Role.USER, content=task))

        self.state.messages = [{"role": m.role.value, "content": m.content} for m in messages]
        tool_schemas = self._get_tool_schemas() if self._tools else None

        # Reasoning loop
        for iteration in range(max_iter):
            self.state.iterations = iteration + 1
            await self._emit("iteration_start", {"iteration": iteration + 1})

            try:
                response = await self._llm.complete(messages, tools=tool_schemas)
            except Exception as e:
                self.state.errors.append(f"LLM error: {e}")
                await self._emit("error", {"error": str(e)})
                if iteration < 2:
                    await asyncio.sleep(2 ** iteration)
                    continue
                break

            assistant_msg = response.message
            messages.append(assistant_msg)

            # If no tool calls, agent is done
            if not assistant_msg.tool_calls:
                self.state.status = TaskStatus.COMPLETED
                await self._emit("completed", {"response": assistant_msg.content})
                break

            # Execute tool calls
            tool_results = await self._execute_tools(assistant_msg.tool_calls)
            for result in tool_results:
                messages.append(Message(
                    role=Role.TOOL,
                    content=result.output,
                    tool_call_id=result.tool_call_id,
                ))
                if not result.success:
                    self.state.errors.append(f"Tool error ({result.tool_call_id}): {result.output}")

            await self._emit("tools_executed", {"count": len(tool_results)})
        else:
            self.state.status = TaskStatus.FAILED
            self.state.errors.append("Max iterations reached")

        # Store final messages
        self.state.messages = [{"role": m.role.value, "content": m.content} for m in messages]
        return self.state

    async def run_with_reflection(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> AgentState:
        """Run with self-evaluation and retry on failure."""
        for attempt in range(max_retries):
            state = await self.run(task, context)

            if state.status == TaskStatus.COMPLETED and state.messages:
                last_msg = state.messages[-1].get("content", "")
                evaluation = await self._reflector.evaluate(task, last_msg)

                if evaluation.get("success") and evaluation.get("score", 0) >= 7:
                    await self._emit("reflection_passed", evaluation)
                    return state

                # Retry with feedback
                await self._emit("reflection_failed", evaluation)
                feedback = evaluation.get("feedback", "Try again")
                context = context or {}
                context["previous_attempt_feedback"] = feedback
                context["attempt"] = attempt + 1
            elif state.status == TaskStatus.FAILED and attempt < max_retries - 1:
                # Replan on failure
                error = state.errors[-1] if state.errors else "Unknown error"
                new_plan = await self._planner.replan(
                    task, [], state.plan[state.current_step] if state.plan else task, error
                )
                context = context or {}
                context["revised_plan"] = new_plan

        return state

    async def _execute_tools(self, tool_calls: list[dict[str, Any]]) -> list[ToolResult]:
        """Execute tool calls with timeout and error handling."""
        results = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            tool_id = tc.get("id", "")

            try:
                args_str = fn.get("arguments", "{}")
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                results.append(ToolResult(tool_call_id=tool_id, output="Invalid JSON arguments", success=False))
                continue

            tool = self._tools.get(tool_name)
            if not tool:
                results.append(ToolResult(tool_call_id=tool_id, output=f"Unknown tool: {tool_name}", success=False))
                continue

            await self._emit("tool_start", {"tool": tool_name, "args": args})

            try:
                result = await asyncio.wait_for(
                    tool.execute(**args),
                    timeout=self._settings.timeout,
                )
                results.append(result if isinstance(result, ToolResult) else ToolResult(
                    tool_call_id=tool_id, output=str(result), success=True
                ))
            except asyncio.TimeoutError:
                results.append(ToolResult(tool_call_id=tool_id, output=f"Tool '{tool_name}' timed out", success=False))
            except Exception as e:
                results.append(ToolResult(tool_call_id=tool_id, output=f"Tool error: {e}", success=False))

            await self._emit("tool_end", {"tool": tool_name, "success": results[-1].success})

        return results

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        await self._event_bus.publish(Event(
            type=f"agent.{event_type}",
            data=data,
            source=self.state.session_id if self.state else "",
        ))
