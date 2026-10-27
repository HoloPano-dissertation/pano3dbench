from __future__ import annotations

from abc import ABC, abstractmethod

from pano3dbench.registry import ComponentRegistry
from pano3dbench.scene import SceneOutput


class Metric(ABC):
    name: str
    requires_ground_truth: bool = True
    dataset_level: bool = False

    @abstractmethod
    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float: ...


METRICS: ComponentRegistry[Metric] = ComponentRegistry(
    "metric", keyword_parameters=False
)
