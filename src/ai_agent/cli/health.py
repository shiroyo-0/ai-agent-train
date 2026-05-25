"""Health check command - verify system configuration."""

import asyncio
import shutil
from typing import Any

from rich.console import Console
from rich.table import Table

from ai_agent.core import get_settings

console = Console()


async def run_health_check() -> None:
    """Run comprehensive health checks."""
    settings = get_settings()
    checks: list[tuple[str, bool, str]] = []

    # Python version
    import sys
    py_ok = sys.version_info >= (3, 12)
    checks.append(("Python 3.12+", py_ok, f"{sys.version_info.major}.{sys.version_info.minor}"))

    # Data directory
    data_ok = settings.data_dir.exists()
    checks.append(("Data directory", data_ok, str(settings.data_dir)))

    # Git
    git_ok = shutil.which("git") is not None
    checks.append(("Git", git_ok, shutil.which("git") or "not found"))

    # Docker
    docker_ok = shutil.which("docker") is not None
    checks.append(("Docker", docker_ok, shutil.which("docker") or "not found"))

    # LLM Provider keys
    has_key = any([
        settings.openai_api_key, settings.anthropic_api_key,
        settings.groq_api_key, settings.openrouter_api_key,
    ])
    checks.append(("LLM API Key", has_key, "configured" if has_key else "no keys set"))

    # Ollama
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = r.status_code == 200
            model_count = len(r.json().get("models", []))
            checks.append(("Ollama", ollama_ok, f"{model_count} models available"))
    except Exception:
        checks.append(("Ollama", False, "not reachable"))

    # Redis
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        checks.append(("Redis", True, settings.redis_url))
        await r.close()
    except Exception:
        checks.append(("Redis", False, "not reachable"))

    # PostgreSQL
    try:
        import asyncpg
        conn = await asyncio.wait_for(
            asyncpg.connect(settings.database_url.replace("+asyncpg", "")), timeout=3
        )
        await conn.close()
        checks.append(("PostgreSQL", True, "connected"))
    except Exception:
        checks.append(("PostgreSQL", False, "not reachable"))

    # ChromaDB
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"http://{settings.chroma_host}:{settings.chroma_port}/api/v1/heartbeat")
            checks.append(("ChromaDB", r.status_code == 200, "connected"))
    except Exception:
        checks.append(("ChromaDB", False, "not reachable"))

    # Display results
    table = Table(title="🏥 System Health Check", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Details", style="dim")

    for name, ok, detail in checks:
        status = "[green]✓ OK[/green]" if ok else "[red]✗ FAIL[/red]"
        table.add_row(name, status, detail)

    console.print(table)

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    console.print(f"\n[{'green' if passed == total else 'yellow'}]{passed}/{total} checks passed[/]")
