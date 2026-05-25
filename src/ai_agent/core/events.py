"""Async event bus for decoupled communication."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, TypeAlias

EventHandler: TypeAlias = Callable[["Event"], Coroutine[Any, Any, None]]


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""


class EventBus:
    """Async pub/sub event bus."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].remove(handler)

    async def publish(self, event: Event) -> None:
        self._history.append(event)
        handlers = self._handlers.get(event.type, []) + self._handlers.get("*", [])
        await asyncio.gather(*(h(event) for h in handlers), return_exceptions=True)

    @property
    def history(self) -> list[Event]:
        return self._history.copy()


_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
