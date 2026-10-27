from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

from pano3dbench.registry import ComponentRegistry
from pano3dbench.scene import SceneOutput, SceneSample


class PredictionSource(ABC):
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    @abstractmethod
    def predict(self, sample: SceneSample) -> SceneOutput: ...

    def predict_many(self, samples: Sequence[SceneSample]) -> Iterable[SceneOutput]:
        for sample in samples:
            yield self.predict(sample)

    def provenance(self) -> dict[str, Any]:
        return {}


PREDICTION_SOURCES: ComponentRegistry[PredictionSource] = ComponentRegistry(
    "prediction source"
)
