"""CLI task runner - executes tasks with progress display."""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.markdown import Markdown

from ai_agent.agents import AgentEngine, MultiAgentOrchestrator
from ai_agent.core import Event, TaskStatus, get_event_bus
from ai_agent.models import LLMClient
from ai_agent.tools import create_default_registry

console = Console()


async def run_task(
    task: str,
    model: str | None = None,
    plan_first: bool = True,
    reflect: bool = False,
) -> None:
    """Execute a task with live progress display."""
    llm = LLMClient(model=model)
    registry = create_default_registry()
    engine = AgentEngine(llm=llm, tools=registry.as_dict)
    event_bus = get_event_bus()

    console.print(Panel(f"[bold]{task}[/bold]", title="🚀 Task", border_style="blue"))

    iteration_count = 0
    tool_count = 0

    async def on_iteration(event: Event) -> None:
        nonlocal iteration_count
        iteration_count = event.data.get("iteration", 0)

    async def on_tool(event: Event) -> None:
        nonlocal tool_count
        tool_count += 1
        tool_name = event.data.get("tool", "")
        console.print(f"  [blue]⚡ {tool_name}[/blue]")

    async def on_plan(event: Event) -> None:
        plan = event.data.get("plan", [])
        console.print("\n[bold cyan]📋 Plan:[/bold cyan]")
        for i, step in enumerate(plan, 1):
            console.print(f"  {i}. {step}")
        console.print()

    event_bus.subscribe("agent.iteration_start", on_iteration)
    event_bus.subscribe("agent.tool_start", on_tool)
    event_bus.subscribe("agent.plan_created", on_plan)

    with console.status("[bold green]Agent working...", spinner="dots"):
        if reflect:
            state = await engine.run_with_reflection(task)
        else:
            state = await engine.run(task, plan_first=plan_first)

    # Display result
    if state.status == TaskStatus.COMPLETED:
        response = ""
        for msg in reversed(state.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                response = msg["content"]
                break
        console.print(Panel(Markdown(response) if "```" in response else response, title="✅ Result", border_style="green"))
    else:
        console.print(Panel(
            "\n".join(state.errors) or "Task failed",
            title="❌ Failed", border_style="red",
        ))

    # Stats
    usage = llm.total_usage
    console.print(f"\n[dim]Iterations: {iteration_count} | Tools used: {tool_count} | Tokens: {usage.total_tokens:,} | Cost: ${usage.cost:.4f}[/dim]")


async def create_plan(task: str, model: str | None = None) -> None:
    """Create and display an execution plan."""
    from ai_agent.agents.engine import Planner
    llm = LLMClient(model=model)
    planner = Planner(llm)

    with console.status("[bold]Planning...", spinner="dots"):
        steps = await planner.create_plan(task)

    console.print(Panel(f"[bold]{task}[/bold]", title="📋 Plan", border_style="cyan"))
    for i, step in enumerate(steps, 1):
        console.print(f"  [cyan]{i}.[/cyan] {step}")
    console.print(f"\n[dim]{len(steps)} steps planned[/dim]")


async def run_multi_agent(task: str, strategy: str = "sequential") -> None:
    """Run multi-agent collaboration with progress display."""
    llm = LLMClient()
    orchestrator = MultiAgentOrchestrator(llm=llm)
    event_bus = get_event_bus()

    console.print(Panel(f"[bold]{task}[/bold]", title="👥 Multi-Agent Task", border_style="magenta"))

    async def on_subtask(event: Event) -> None:
        agent = event.data.get("agent", "")
        console.print(f"  [magenta]🔄 {agent} agent working...[/magenta]")

    event_bus.subscribe("multi_agent.subtask_start", on_subtask)

    with console.status("[bold magenta]Agents collaborating...", spinner="dots"):
        result = await orchestrator.execute(task, strategy=strategy)

    # Display results
    table = Table(title="Subtask Results")
    table.add_column("Agent", style="cyan")
    table.add_column("Task")
    table.add_column("Status", style="green")
    for st in result.subtasks:
        status_icon = "✅" if st.status == TaskStatus.COMPLETED else "❌"
        table.add_row(st.agent_role.value, st.task[:50], status_icon)
    console.print(table)

    console.print(Panel(
        Markdown(result.final_result) if "```" in result.final_result else result.final_result,
        title="📝 Final Result", border_style="green" if result.success else "red",
    ))
