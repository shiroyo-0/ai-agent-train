"""Workflow engine - DAG-based task automation with conditions and parallel execution."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

from ai_agent.core import Event, TaskStatus, get_event_bus, get_logger, get_settings

logger = get_logger(__name__)


class NodeType(str, Enum):
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"


@dataclass
class WorkflowNode:
    id: str
    name: str
    node_type: NodeType = NodeType.TASK
    action: str = ""  # agent task or shell command
    depends_on: list[str] = field(default_factory=list)
    condition: str = ""  # for condition nodes
    config: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    duration: float = 0.0


@dataclass
class Workflow:
    id: str
    name: str
    description: str = ""
    nodes: list[WorkflowNode] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """Execute DAG-based workflows with dependency resolution."""

    def __init__(self, executor: Callable[..., Coroutine] | None = None) -> None:
        self._event_bus = get_event_bus()
        self._settings = get_settings()
        self._workflows_dir = self._settings.data_dir / "workflows"
        self._workflows_dir.mkdir(parents=True, exist_ok=True)
        self._executor = executor  # function(action, context) -> result
        self._running: dict[str, Workflow] = {}

    async def execute(self, workflow: Workflow) -> Workflow:
        """Execute a workflow respecting dependencies."""
        workflow.status = TaskStatus.RUNNING
        self._running[workflow.id] = workflow
        await self._emit("workflow_start", {"id": workflow.id, "name": workflow.name})

        try:
            completed: set[str] = set()
            while True:
                # Find ready nodes (all deps satisfied)
                ready = [
                    n for n in workflow.nodes
                    if n.status == TaskStatus.PENDING and all(d in completed for d in n.depends_on)
                ]
                if not ready:
                    break

                # Execute ready nodes (parallel if multiple)
                tasks = [self._execute_node(n, workflow.context) for n in ready]
                await asyncio.gather(*tasks)

                for n in ready:
                    if n.status == TaskStatus.COMPLETED:
                        completed.add(n.id)
                        workflow.context[f"result_{n.id}"] = n.result
                    elif n.status == TaskStatus.FAILED:
                        workflow.status = TaskStatus.FAILED
                        workflow.completed_at = time.time()
                        self._save(workflow)
                        return workflow

            # Check if all nodes completed
            all_done = all(n.status == TaskStatus.COMPLETED for n in workflow.nodes)
            workflow.status = TaskStatus.COMPLETED if all_done else TaskStatus.FAILED
        except Exception as e:
            workflow.status = TaskStatus.FAILED
            logger.error("workflow_failed", id=workflow.id, error=str(e))

        workflow.completed_at = time.time()
        self._save(workflow)
        await self._emit("workflow_complete", {"id": workflow.id, "status": workflow.status.value})
        return workflow

    async def _execute_node(self, node: WorkflowNode, context: dict[str, Any]) -> None:
        """Execute a single workflow node."""
        node.status = TaskStatus.RUNNING
        start = time.time()

        try:
            if node.node_type == NodeType.CONDITION:
                # Evaluate condition
                result = eval(node.condition, {"ctx": context})  # noqa: S307
                node.result = str(result)
            elif self._executor:
                node.result = await self._executor(node.action, context)
            else:
                node.result = f"No executor configured for: {node.action}"

            node.status = TaskStatus.COMPLETED
        except Exception as e:
            node.status = TaskStatus.FAILED
            node.error = str(e)

        node.duration = time.time() - start

    def create_workflow(self, name: str, steps: list[dict[str, Any]]) -> Workflow:
        """Create a workflow from a list of step definitions."""
        nodes = []
        for i, step in enumerate(steps):
            node = WorkflowNode(
                id=step.get("id", f"step_{i}"),
                name=step.get("name", f"Step {i+1}"),
                node_type=NodeType(step.get("type", "task")),
                action=step.get("action", ""),
                depends_on=step.get("depends_on", []),
                condition=step.get("condition", ""),
                config=step.get("config", {}),
            )
            nodes.append(node)

        return Workflow(id=str(uuid.uuid4())[:8], name=name, nodes=nodes)

    def load_workflow(self, path: str | Path) -> Workflow:
        """Load workflow definition from YAML/JSON file."""
        p = Path(path)
        data = json.loads(p.read_text())
        return self.create_workflow(data["name"], data["steps"])

    def list_workflows(self) -> list[dict[str, Any]]:
        """List saved workflow results."""
        results = []
        for f in self._workflows_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                results.append({"id": data["id"], "name": data["name"], "status": data["status"]})
            except Exception:
                pass
        return results

    def _save(self, workflow: Workflow) -> None:
        data = {
            "id": workflow.id, "name": workflow.name, "status": workflow.status.value,
            "created_at": workflow.created_at, "completed_at": workflow.completed_at,
            "nodes": [
                {"id": n.id, "name": n.name, "status": n.status.value, "result": n.result[:500], "duration": n.duration}
                for n in workflow.nodes
            ],
        }
        (self._workflows_dir / f"{workflow.id}.json").write_text(json.dumps(data, indent=2))

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        await self._event_bus.publish(Event(type=f"workflow.{event_type}", data=data))
