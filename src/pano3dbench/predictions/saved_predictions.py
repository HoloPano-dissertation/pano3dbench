from __future__ import annotations

import json
from pathlib import Path

from pano3dbench.predictions.base import PREDICTION_SOURCES, PredictionSource
from pano3dbench.scene import SceneOutput, SceneSample


@PREDICTION_SOURCES.register("fixture")
class SavedPredictionSource(PredictionSource):
    def __init__(self, prediction_dir: str | Path, seed: int | None = None) -> None:
        super().__init__(seed)
        self.prediction_dir = Path(prediction_dir)

    def predict(self, sample: SceneSample) -> SceneOutput:
        prediction_path = self.prediction_dir / f"{sample.sample_id}.json"
        with prediction_path.open(encoding="utf-8") as prediction_file:
            return SceneOutput.from_dict(json.load(prediction_file))
