"""Dependency injection container."""

from typing import Any, TypeVar, get_type_hints

T = TypeVar("T")


class Container:
    """Simple DI container with singleton and factory support."""

    def __init__(self) -> None:
        self._singletons: dict[type, Any] = {}
        self._factories: dict[type, Any] = {}

    def register_singleton(self, interface: type[T], instance: T) -> None:
        self._singletons[interface] = instance

    def register_factory(self, interface: type[T], factory: Any) -> None:
        self._factories[interface] = factory

    def resolve(self, interface: type[T]) -> T:
        if interface in self._singletons:
            return self._singletons[interface]
        if interface in self._factories:
            instance = self._factories[interface]()
            self._singletons[interface] = instance
            return instance
        raise KeyError(f"No registration for {interface}")

    def has(self, interface: type) -> bool:
        return interface in self._singletons or interface in self._factories


_container: Container | None = None


def get_container() -> Container:
    global _container
    if _container is None:
        _container = Container()
    return _container
