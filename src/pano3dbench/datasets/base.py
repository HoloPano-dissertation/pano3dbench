from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from pano3dbench.registry import ComponentRegistry
from pano3dbench.scene import SceneSample


class DatasetLoader(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[SceneSample]: ...


DATASET_LOADERS: ComponentRegistry[DatasetLoader] = ComponentRegistry("dataset loader")
