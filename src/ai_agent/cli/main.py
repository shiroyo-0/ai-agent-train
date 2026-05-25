"""AI Agent CLI - Modern terminal interface with Rich/Typer."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red bold",
    "agent": "magenta bold",
    "tool": "blue",
})

console = Console(theme=custom_theme)
app = typer.Typer(
    name="ai",
    help="🤖 AI Coding Agent - Your intelligent development companion",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _run_async(coro):
    """Run async function from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send to the agent"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    no_tools: bool = typer.Option(False, "--no-tools", help="Disable tool use"),
):
    """💬 Start an interactive chat session with the AI agent."""
    from ai_agent.cli.session import ChatSession
    session = ChatSession(model=model, tools_enabled=not no_tools)
    if message:
        _run_async(session.send(message))
    else:
        _run_async(session.interactive())


@app.command()
def run(
    task: str = typer.Argument(..., help="Task for the agent to execute"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    plan: bool = typer.Option(True, "--plan/--no-plan", help="Create plan before executing"),
    reflect: bool = typer.Option(False, "--reflect", "-r", help="Enable self-reflection"),
):
    """🚀 Execute a task autonomously."""
    from ai_agent.cli.runner import run_task
    _run_async(run_task(task, model=model, plan_first=plan, reflect=reflect))


@app.command()
def plan(
    task: str = typer.Argument(..., help="Task to plan"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """📋 Create an execution plan for a task."""
    from ai_agent.cli.runner import create_plan
    _run_async(create_plan(task, model=model))


@app.command()
def init(
    path: Path = typer.Argument(Path("."), help="Project directory"),
):
    """🏗️  Initialize AI agent for a project (analyze repo)."""
    from ai_agent.repository import RepositoryAnalyzer
    console.print(f"\n[info]Analyzing repository at {path.resolve()}...[/info]\n")
    analyzer = RepositoryAnalyzer(path)
    analysis = analyzer.analyze()

    table = Table(title="📊 Repository Analysis", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Files", str(analysis.file_count))
    table.add_row("Lines", f"{analysis.total_lines:,}")
    table.add_row("Languages", ", ".join(f"{k} ({v})" for k, v in list(analysis.languages.items())[:5]))
    table.add_row("Frameworks", ", ".join(analysis.frameworks) or "None detected")
    table.add_row("Architecture", analysis.architecture)
    table.add_row("Entrypoints", ", ".join(analysis.entrypoints[:3]) or "None found")
    console.print(table)

    if analysis.dependencies:
        dep_table = Table(title="📦 Dependencies")
        dep_table.add_column("Ecosystem")
        dep_table.add_column("Packages")
        for eco, pkgs in analysis.dependencies.items():
            dep_table.add_row(eco, ", ".join(pkgs[:10]) + (f" (+{len(pkgs)-10})" if len(pkgs) > 10 else ""))
        console.print(dep_table)

    console.print(f"\n[success]✓ {analysis.summary}[/success]\n")


@app.command()
def memory(
    action: str = typer.Argument("stats", help="Action: stats, search, clear"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search query"),
):
    """🧠 Manage agent memory."""
    from ai_agent.memory import MemoryManager
    mgr = MemoryManager()

    if action == "stats":
        stats = mgr.stats
        table = Table(title="🧠 Memory Stats")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="white")
        for k, v in stats.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)
    elif action == "search" and query:
        results = mgr.recall(query, limit=10)
        if results:
            for r in results:
                console.print(Panel(r.content[:200], title=f"[{r.memory_type.value}] score={r.score:.2f}"))
        else:
            console.print("[warning]No memories found.[/warning]")
    elif action == "clear":
        if typer.confirm("Clear all memory?"):
            path = mgr._persist_dir / "memory.json"
            path.unlink(missing_ok=True)
            console.print("[success]Memory cleared.[/success]")


@app.command()
def tools():
    """🔧 List available tools."""
    from ai_agent.tools import create_default_registry
    registry = create_default_registry()
    table = Table(title="🔧 Available Tools", show_header=True)
    table.add_column("Tool", style="cyan bold")
    table.add_column("Description", style="white")
    table.add_column("Permissions", style="yellow")
    for tool in registry.list_tools():
        perms = ", ".join(p.value for p in tool.permissions)
        table.add_row(tool.name, tool.description[:60], perms)
    console.print(table)


@app.command()
def models():
    """🤖 List available models."""
    from ai_agent.models import list_models
    table = Table(title="🤖 Available Models", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Provider", style="blue")
    table.add_column("Context", style="white")
    table.add_column("Tools", style="green")
    for m in list_models():
        table.add_row(m.id, m.provider, f"{m.context_window:,}", "✓" if m.supports_tools else "✗")
    console.print(table)


@app.command()
def doctor():
    """🏥 Check system health and configuration."""
    from ai_agent.cli.health import run_health_check
    _run_async(run_health_check())


@app.command()
def agents(
    task: Optional[str] = typer.Argument(None, help="Task for multi-agent execution"),
    strategy: str = typer.Option("sequential", "--strategy", "-s", help="Execution strategy: sequential or parallel"),
):
    """👥 Run multi-agent collaboration."""
    if not task:
        table = Table(title="👥 Available Agents")
        table.add_column("Role", style="cyan bold")
        table.add_column("Description")
        roles = {
            "Manager": "Decomposes tasks, delegates, synthesizes results",
            "Coder": "Writes production-quality code",
            "Reviewer": "Reviews code for bugs and best practices",
            "Debugger": "Analyzes errors and implements fixes",
            "Researcher": "Investigates codebases and patterns",
            "Planner": "Creates implementation plans",
        }
        for role, desc in roles.items():
            table.add_row(role, desc)
        console.print(table)
        return

    from ai_agent.cli.runner import run_multi_agent
    _run_async(run_multi_agent(task, strategy=strategy))


@app.command()
def train(
    action: str = typer.Argument("status", help="Action: generate, finetune, evaluate, status"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """🎓 Training pipeline management."""
    from ai_agent.cli.training_cmd import handle_training
    _run_async(handle_training(action, dataset=dataset, model_name=model_name))


@app.command()
def benchmark(
    suite: str = typer.Option("basic", "--suite", "-s", help="Benchmark suite to run"),
):
    """📊 Run performance benchmarks."""
    console.print(f"[info]Running benchmark suite: {suite}[/info]")
    console.print("[warning]Benchmark system ready. Configure test cases in configs/benchmarks/[/warning]")


@app.command()
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
):
    """📜 View agent logs."""
    from ai_agent.core import get_settings
    log_dir = get_settings().data_dir / "logs"
    log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not log_files:
        console.print("[warning]No log files found.[/warning]")
        return
    latest = log_files[0]
    content = latest.read_text().splitlines()[-lines:]
    for line in content:
        console.print(line)


@app.command()
def rag(
    action: str = typer.Argument("status", help="Action: ingest, query, status, clear"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="File/directory to ingest"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Query to search"),
):
    """📚 RAG knowledge base management."""
    from ai_agent.rag import RAGPipeline
    pipeline = RAGPipeline()

    if action == "ingest" and path:
        p = Path(path)
        with console.status(f"[bold]Ingesting {p}..."):
            if p.is_dir():
                count = pipeline.ingest_directory(p)
            else:
                count = pipeline.ingest_file(p)
        console.print(f"[success]✓ Ingested {count} chunks from {p}[/success]")
    elif action == "query" and query:
        results = pipeline.query(query)
        if results:
            for i, r in enumerate(results, 1):
                source = Path(r["source"]).name if r["source"] else "?"
                console.print(f"\n[cyan][{i}][/cyan] ({source}) score={r.get('score', 0):.2f}")
                console.print(f"  {r['content'][:200]}")
        else:
            console.print("[warning]No results found.[/warning]")
    elif action == "clear":
        if typer.confirm("Clear RAG knowledge base?"):
            pipeline.clear()
            console.print("[success]RAG cleared.[/success]")
    else:
        stats = pipeline.stats
        console.print(f"[info]📚 RAG: {stats['documents']} chunks from {stats['sources']} sources[/info]")


@app.command()
def plugins(
    action: str = typer.Argument("list", help="Action: list, load, unload"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Plugin name"),
):
    """🔌 Plugin management."""
    from ai_agent.plugins import PluginRegistry
    registry = PluginRegistry()

    if action == "list":
        available = registry.list_available()
        loaded = registry.list_loaded()
        if available:
            table = Table(title="🔌 Plugins")
            table.add_column("Name", style="cyan")
            table.add_column("Version")
            table.add_column("Status")
            table.add_column("Description")
            for p in available:
                status = "[green]loaded[/green]" if p.name in loaded else "[dim]available[/dim]"
                table.add_row(p.name, p.version, status, p.description)
            console.print(table)
        else:
            console.print("[dim]No plugins found. Add plugins to ~/.ai-agent/plugins/[/dim]")
    elif action == "load" and name:
        _run_async(registry.load(name))
        console.print(f"[success]Plugin '{name}' loaded.[/success]")
    elif action == "unload" and name:
        _run_async(registry.unload(name))
        console.print(f"[success]Plugin '{name}' unloaded.[/success]")


@app.command()
def workflow(
    action: str = typer.Argument("list", help="Action: run, list, create"),
    path: Optional[str] = typer.Option(None, "--file", "-f", help="Workflow definition file"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Workflow name"),
):
    """⚡ Workflow automation."""
    from ai_agent.workflows import WorkflowEngine
    engine = WorkflowEngine()

    if action == "list":
        workflows = engine.list_workflows()
        if workflows:
            table = Table(title="⚡ Workflows")
            table.add_column("ID", style="cyan")
            table.add_column("Name")
            table.add_column("Status")
            for w in workflows:
                table.add_row(w["id"], w["name"], w["status"])
            console.print(table)
        else:
            console.print("[dim]No workflows yet.[/dim]")
    elif action == "run" and path:
        wf = engine.load_workflow(path)
        console.print(f"[info]Running workflow: {wf.name}[/info]")
        result = _run_async(engine.execute(wf))
        status = "[green]✓[/green]" if result.status.value == "completed" else "[red]✗[/red]"
        console.print(f"{status} Workflow {result.status.value} ({len(result.nodes)} steps)")


@app.command(name="shell")
def agent_shell(
    command: Optional[str] = typer.Argument(None, help="Command to run with AI assistance"),
):
    """🐚 AI-powered shell (agent interprets and executes)."""
    if command:
        from ai_agent.cli.runner import run_task
        _run_async(run_task(f"Execute this shell task: {command}", plan_first=False))
    else:
        console.print("[info]🐚 AI Shell - type commands, agent will help execute[/info]")
        console.print("[dim]Type 'exit' to quit[/dim]\n")
        while True:
            try:
                cmd = input("ai-shell❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd in ("exit", "quit"):
                break
            if not cmd:
                continue
            from ai_agent.cli.runner import run_task
            _run_async(run_task(f"Execute: {cmd}", plan_first=False))


@app.command()
def sandbox(
    command: str = typer.Argument(..., help="Command to run in sandbox"),
    image: str = typer.Option("python:3.12-slim", "--image", "-i", help="Docker image"),
):
    """📦 Run command in isolated Docker sandbox."""
    import asyncio
    from ai_agent.tools.sandbox import DockerSandboxTool
    tool = DockerSandboxTool()
    result = _run_async(tool.execute(command=command, image=image))
    if result.success:
        console.print(result.output)
    else:
        console.print(f"[error]{result.output}[/error]")


@app.callback()
def main_callback():
    """🤖 AI Coding Agent - Intelligent development companion."""
    pass


def version_callback(value: bool):
    if value:
        from ai_agent import __version__
        console.print(f"AI Agent v{__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
