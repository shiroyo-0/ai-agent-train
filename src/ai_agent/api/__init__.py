"""FastAPI REST API for the AI Agent."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai_agent.agents import AgentEngine, MultiAgentOrchestrator
from ai_agent.core import TaskStatus, get_settings
from ai_agent.memory import MemoryManager, MemoryType
from ai_agent.models import LLMClient, list_models
from ai_agent.tools import create_default_registry


# --- Request/Response Models ---

class ChatRequest(BaseModel):
    message: str
    model: str | None = None
    tools_enabled: bool = True
    context: dict[str, Any] = {}

class TaskRequest(BaseModel):
    task: str
    model: str | None = None
    plan_first: bool = True
    reflect: bool = False
    max_iterations: int = 50

class MultiAgentRequest(BaseModel):
    task: str
    strategy: str = "sequential"

class MemoryStoreRequest(BaseModel):
    content: str
    memory_type: str = "long_term"
    tags: list[str] = []

class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 5

class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: str = ""
    errors: list[str] = []
    iterations: int = 0

class HealthResponse(BaseModel):
    status: str
    version: str
    model: str


# --- App State ---

_tasks: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup/shutdown."""
    settings = get_settings()
    settings.ensure_dirs()
    yield


# --- App ---

app = FastAPI(
    title="AI Coding Agent API",
    description="REST API for the AI Coding Agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    settings = get_settings()
    return HealthResponse(status="ok", version="0.1.0", model=settings.default_model)


@app.post("/chat", response_model=TaskResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response."""
    llm = LLMClient(model=request.model)
    registry = create_default_registry() if request.tools_enabled else None
    engine = AgentEngine(llm=llm, tools=registry.as_dict if registry else {})

    state = await engine.run(task=request.message, context=request.context, plan_first=False)

    response = ""
    for msg in reversed(state.messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            response = msg["content"]
            break

    return TaskResponse(
        task_id=state.session_id,
        status=state.status.value,
        result=response,
        errors=state.errors,
        iterations=state.iterations,
    )


@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Create and execute a task asynchronously."""
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "running", "result": "", "errors": []}

    async def _execute():
        llm = LLMClient(model=request.model)
        registry = create_default_registry()
        engine = AgentEngine(llm=llm, tools=registry.as_dict)
        if request.reflect:
            state = await engine.run_with_reflection(request.task)
        else:
            state = await engine.run(request.task, plan_first=request.plan_first, max_iterations=request.max_iterations)
        response = ""
        for msg in reversed(state.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                response = msg["content"]
                break
        _tasks[task_id] = {"status": state.status.value, "result": response, "errors": state.errors, "iterations": state.iterations}

    background_tasks.add_task(_execute)
    return TaskResponse(task_id=task_id, status="running")


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get task status and result."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    t = _tasks[task_id]
    return TaskResponse(task_id=task_id, status=t["status"], result=t.get("result", ""), errors=t.get("errors", []), iterations=t.get("iterations", 0))


@app.post("/agents/multi")
async def multi_agent(request: MultiAgentRequest):
    """Run multi-agent collaboration."""
    llm = LLMClient()
    orchestrator = MultiAgentOrchestrator(llm=llm)
    result = await orchestrator.execute(request.task, strategy=request.strategy)
    return {
        "task": result.task,
        "success": result.success,
        "final_result": result.final_result,
        "subtasks": [{"agent": st.agent_role.value, "task": st.task, "status": st.status.value} for st in result.subtasks],
    }


@app.post("/memory/store")
async def memory_store(request: MemoryStoreRequest):
    """Store a memory entry."""
    mgr = MemoryManager()
    entry_id = mgr.store(request.content, MemoryType(request.memory_type), tags=request.tags)
    mgr.persist()
    return {"id": entry_id, "stored": True}


@app.post("/memory/search")
async def memory_search(request: MemorySearchRequest):
    """Search memory."""
    mgr = MemoryManager()
    results = mgr.recall(request.query, limit=request.limit)
    return {"results": [{"id": r.id, "content": r.content, "score": r.score, "type": r.memory_type.value} for r in results]}


@app.get("/memory/stats")
async def memory_stats():
    """Get memory statistics."""
    mgr = MemoryManager()
    return mgr.stats


@app.get("/models")
async def get_models():
    """List available models."""
    return {"models": [{"id": m.id, "provider": m.provider, "name": m.name, "context_window": m.context_window} for m in list_models()]}


@app.get("/tools")
async def get_tools():
    """List available tools."""
    registry = create_default_registry()
    return {"tools": [{"name": t.name, "description": t.description, "permissions": [p.value for p in t.permissions]} for t in registry.list_tools()]}


def start_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
