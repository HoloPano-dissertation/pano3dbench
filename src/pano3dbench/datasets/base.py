from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from pano3dbench.domain import SceneSample
from pano3dbench.registry import ComponentRegistry


class BaseDatasetLoader(ABC):
    registry: ComponentRegistry[BaseDatasetLoader] = ComponentRegistry(
        "dataset loader"
    )

    @abstractmethod
    def samples(self) -> Iterator[SceneSample]:
        """Yield normalized samples for the benchmark."""

    def __iter__(self) -> Iterator[SceneSample]:
        return self.samples()

    @classmethod
    def create(
        cls, name: str, parameters: dict[str, Any] | None = None
    ) -> BaseDatasetLoader:
        return cls.registry.create(name, parameters)
