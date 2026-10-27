from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ComponentRegistry(Generic[T]):
    def __init__(self, component_name: str, *, keyword_parameters: bool = True) -> None:
        self.component_name = component_name
        self.keyword_parameters = keyword_parameters
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[F], F]:
        normalized = name.strip().lower()

        def decorator(factory: F) -> F:
            if normalized in self._factories:
                raise ValueError(
                    f"{self.component_name} '{normalized}' is already registered"
                )
            self._factories[normalized] = factory
            return factory

        return decorator

    def alias(self, name: str, *aliases: str) -> None:
        normalized = name.strip().lower()
        if normalized not in self._factories:
            raise ValueError(
                f"Cannot alias unknown {self.component_name} '{name}'; "
                f"register it first"
            )
        for candidate in aliases:
            alias_name = candidate.strip().lower()
            if alias_name in self._factories:
                raise ValueError(
                    f"{self.component_name} '{alias_name}' is already registered"
                )
            self._factories[alias_name] = self._factories[normalized]

    def create(self, name: str, parameters: dict[str, Any] | None = None) -> T:
        factory = self._factory(name)
        parameters = dict(parameters or {})
        if not self.keyword_parameters:
            return factory(parameters)
        try:
            inspect.signature(factory).bind(**parameters)
        except TypeError as error:
            raise ValueError(
                f"Invalid parameters for {self.component_name} '{name}': {error}"
            ) from None
        return factory(**parameters)

    def _factory(self, name: str) -> Callable[..., T]:
        normalized = name.strip().lower()
        try:
            return self._factories[normalized]
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise ValueError(
                f"Unknown {self.component_name} '{name}'. Available: {available}"
            ) from error

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
