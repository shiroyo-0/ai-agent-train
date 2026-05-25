"""Interactive chat session with streaming and slash commands."""

import asyncio
import time
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax

from ai_agent.agents import AgentEngine
from ai_agent.core import Event, Message, Role, get_event_bus, get_settings
from ai_agent.memory import MemoryManager, MemoryType
from ai_agent.models import LLMClient
from ai_agent.tools import create_default_registry

console = Console()

SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/clear": "Clear conversation history",
    "/model": "Switch model (e.g., /model openai/gpt-4o)",
    "/tools": "Toggle tool use",
    "/memory": "Show memory stats",
    "/save": "Save session",
    "/load": "Load session",
    "/plan": "Create a plan for a task",
    "/multi": "Use multi-agent mode",
    "/exit": "Exit chat",
}


class ChatSession:
    """Interactive chat session with the AI agent."""

    def __init__(self, model: str | None = None, tools_enabled: bool = True) -> None:
        self._settings = get_settings()
        self._llm = LLMClient(model=model)
        self._tools_enabled = tools_enabled
        self._registry = create_default_registry() if tools_enabled else None
        self._engine = AgentEngine(
            llm=self._llm,
            tools=self._registry.as_dict if self._registry else {},
        )
        self._memory = MemoryManager()
        self._messages: list[Message] = []
        self._event_bus = get_event_bus()
        self._session_start = time.time()

        # Subscribe to events for live display
        self._event_bus.subscribe("agent.tool_start", self._on_tool_start)
        self._event_bus.subscribe("agent.tool_end", self._on_tool_end)

    async def interactive(self) -> None:
        """Run interactive chat loop."""
        history_path = self._settings.data_dir / "sessions" / "chat_history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_session: PromptSession = PromptSession(history=FileHistory(str(history_path)))

        console.print(Panel(
            "[bold magenta]🤖 AI Coding Agent[/bold magenta]\n"
            f"Model: [cyan]{self._llm.model}[/cyan] | Tools: [{'green' if self._tools_enabled else 'red'}]{'enabled' if self._tools_enabled else 'disabled'}[/]\n"
            "Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit",
            title="Welcome", border_style="blue",
        ))

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: prompt_session.prompt("\n❯ ")
                )
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                if await self._handle_slash_command(user_input):
                    continue
                if user_input == "/exit":
                    break

            await self.send(user_input)

        self._save_session()
        console.print("\n[info]Session saved. Goodbye! 👋[/info]")

    async def send(self, message: str) -> str:
        """Send a message and get a response."""
        self._messages.append(Message(role=Role.USER, content=message))

        # Store in short-term memory
        self._memory.store(message, MemoryType.SHORT_TERM, tags=["user_input"])

        # Recall relevant context
        relevant = self._memory.recall(message, limit=3, memory_type=MemoryType.LONG_TERM)
        context = {}
        if relevant:
            context["relevant_memories"] = [r.content for r in relevant]

        console.print()
        with console.status("[bold blue]Thinking...", spinner="dots"):
            state = await self._engine.run(
                task=message,
                context=context,
                plan_first=False,
            )

        # Extract response
        response = ""
        if state.messages:
            for msg in reversed(state.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    response = msg["content"]
                    break

        # Display response
        if response:
            self._display_response(response)
            self._messages.append(Message(role=Role.ASSISTANT, content=response))
            self._memory.store(response, MemoryType.SHORT_TERM, tags=["assistant_response"])

        # Show usage
        usage = self._llm.total_usage
        if usage.total_tokens > 0:
            console.print(
                f"\n[dim]tokens: {usage.total_tokens:,} | cost: ${usage.cost:.4f}[/dim]",
                justify="right",
            )

        return response

    def _display_response(self, response: str) -> None:
        """Display response with syntax highlighting for code blocks."""
        # Check if response contains code blocks
        if "```" in response:
            console.print(Markdown(response))
        else:
            console.print(Panel(response, border_style="green", title="🤖 Agent"))

    async def _handle_slash_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            from rich.table import Table
            table = Table(title="Slash Commands", show_header=True)
            table.add_column("Command", style="cyan")
            table.add_column("Description")
            for cmd_name, desc in SLASH_COMMANDS.items():
                table.add_row(cmd_name, desc)
            console.print(table)
            return True

        elif command == "/clear":
            self._messages.clear()
            console.print("[success]Conversation cleared.[/success]")
            return True

        elif command == "/model":
            if args:
                self._llm = LLMClient(model=args)
                self._engine = AgentEngine(
                    llm=self._llm,
                    tools=self._registry.as_dict if self._registry else {},
                )
                console.print(f"[success]Switched to model: {args}[/success]")
            else:
                console.print(f"[info]Current model: {self._llm.model}[/info]")
            return True

        elif command == "/tools":
            self._tools_enabled = not self._tools_enabled
            status = "enabled" if self._tools_enabled else "disabled"
            console.print(f"[info]Tools {status}[/info]")
            return True

        elif command == "/memory":
            stats = self._memory.stats
            console.print(f"[info]Memory: {stats}[/info]")
            return True

        elif command == "/save":
            self._save_session()
            console.print("[success]Session saved.[/success]")
            return True

        elif command == "/exit":
            return False  # Signal to exit

        return False

    def _save_session(self) -> None:
        """Persist session to disk."""
        import json
        session_dir = self._settings.data_dir / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "messages": [{"role": m.role.value, "content": m.content} for m in self._messages],
            "model": self._llm.model,
            "duration": time.time() - self._session_start,
        }
        session_file = session_dir / f"session_{int(self._session_start)}.json"
        session_file.write_text(json.dumps(data, indent=2))

    async def _on_tool_start(self, event: Event) -> None:
        tool_name = event.data.get("tool", "")
        console.print(f"  [tool]⚡ Using tool: {tool_name}[/tool]")

    async def _on_tool_end(self, event: Event) -> None:
        tool_name = event.data.get("tool", "")
        success = event.data.get("success", False)
        icon = "✓" if success else "✗"
        style = "success" if success else "error"
        console.print(f"  [{style}]{icon} {tool_name} complete[/{style}]")
