"""Plugin system - dynamic loading, registration, and lifecycle management."""

import importlib
import importlib.util
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_agent.core import get_logger, get_settings

logger = get_logger(__name__)


@dataclass
class PluginMeta:
    name: str
    version: str
    description: str
    author: str = ""
    dependencies: list[str] = field(default_factory=list)
    entry_point: str = ""


class BasePlugin(ABC):
    """Base class for all plugins."""

    meta: PluginMeta

    @abstractmethod
    async def activate(self, context: dict[str, Any]) -> None:
        """Called when plugin is loaded."""
        ...

    @abstractmethod
    async def deactivate(self) -> None:
        """Called when plugin is unloaded."""
        ...

    def get_tools(self) -> list[Any]:
        """Return tools this plugin provides."""
        return []

    def get_commands(self) -> dict[str, Any]:
        """Return CLI commands this plugin adds."""
        return {}


class PluginRegistry:
    """Discovers, loads, and manages plugins."""

    def __init__(self, plugin_dir: Path | None = None) -> None:
        self._settings = get_settings()
        self._plugin_dir = plugin_dir or self._settings.data_dir / "plugins"
        self._plugin_dir.mkdir(parents=True, exist_ok=True)
        self._plugins: dict[str, BasePlugin] = {}
        self._meta: dict[str, PluginMeta] = {}

    def discover(self) -> list[PluginMeta]:
        """Discover available plugins in plugin directory."""
        found = []
        for p in self._plugin_dir.iterdir():
            if p.is_dir() and (p / "plugin.json").exists():
                try:
                    meta_data = json.loads((p / "plugin.json").read_text())
                    meta = PluginMeta(**meta_data)
                    self._meta[meta.name] = meta
                    found.append(meta)
                except Exception as e:
                    logger.warning("plugin_discover_failed", path=str(p), error=str(e))
            elif p.suffix == ".py" and p.stem != "__init__":
                meta = PluginMeta(name=p.stem, version="0.1.0", description=f"Plugin: {p.stem}", entry_point=str(p))
                self._meta[meta.name] = meta
                found.append(meta)
        return found

    async def load(self, name: str, context: dict[str, Any] | None = None) -> BasePlugin | None:
        """Load and activate a plugin by name."""
        if name in self._plugins:
            return self._plugins[name]

        meta = self._meta.get(name)
        if not meta:
            self.discover()
            meta = self._meta.get(name)
        if not meta:
            logger.error("plugin_not_found", name=name)
            return None

        try:
            plugin_path = self._plugin_dir / name
            if plugin_path.is_dir():
                module_path = plugin_path / "__init__.py"
            else:
                module_path = self._plugin_dir / f"{name}.py"

            spec = importlib.util.spec_from_file_location(f"plugins.{name}", str(module_path))
            if not spec or not spec.loader:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            plugin_class = getattr(module, "Plugin", None)
            if not plugin_class:
                return None

            plugin = plugin_class()
            await plugin.activate(context or {})
            self._plugins[name] = plugin
            logger.info("plugin_loaded", name=name)
            return plugin
        except Exception as e:
            logger.error("plugin_load_failed", name=name, error=str(e))
            return None

    async def unload(self, name: str) -> bool:
        """Deactivate and unload a plugin."""
        plugin = self._plugins.pop(name, None)
        if plugin:
            await plugin.deactivate()
            return True
        return False

    def list_loaded(self) -> list[str]:
        return list(self._plugins.keys())

    def list_available(self) -> list[PluginMeta]:
        return self.discover()

    def get(self, name: str) -> BasePlugin | None:
        return self._plugins.get(name)
