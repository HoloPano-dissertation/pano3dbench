from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pano3dbench.config import ExperimentConfig
from pano3dbench.datasets import DatasetLoader
from pano3dbench.evaluation.aggregation import aggregate
from pano3dbench.evaluation.result_writer import ResultWriter
from pano3dbench.metrics import Metric
from pano3dbench.predictions import PredictionSource
from pano3dbench.provenance import git_revision
from pano3dbench.scene import SceneOutput

LabelledPrediction = tuple[SceneOutput, SceneOutput]


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    scene_count: int
    summary: dict[str, dict[str, Any]]


class Evaluator:
    def __init__(
        self,
        config: ExperimentConfig,
        dataset: DatasetLoader,
        prediction_source: PredictionSource,
        metrics: list[Metric],
    ) -> None:
        _require_unique_names(metrics)
        self.config = config
        self.dataset = dataset
        self.prediction_source = prediction_source
        self.scene_metrics = [
            metric for metric in metrics if not getattr(metric, "dataset_level", False)
        ]
        self.dataset_metrics = [
            metric for metric in metrics if getattr(metric, "dataset_level", False)
        ]
        self.writer = ResultWriter(config.output_dir)

    def run(self) -> EvaluationResult:
        random.seed(self.config.seed)
        self.writer.prepare()

        values: dict[str, list[float]] = {
            metric.name: [] for metric in self.scene_metrics
        }
        records: list[dict[str, Any]] = []
        labelled: list[LabelledPrediction] = []

        samples = list(self.dataset)
        predictions = self.prediction_source.predict_many(samples)
        for sample, prediction in zip(samples, predictions):
            self.writer.write_prediction(sample.sample_id, prediction)
            scene_values = self._evaluate_scene(prediction, sample.ground_truth)
            for name, value in scene_values.items():
                values[name].append(value)
            records.append({"sample_id": sample.sample_id, "metrics": scene_values})
            if sample.ground_truth is not None:
                labelled.append((prediction, sample.ground_truth))
        self.writer.write_scene_metrics(records)

        summary: dict[str, dict[str, Any]] = {
            name: aggregate(scene_values, self.config.seed)
            for name, scene_values in values.items()
        }
        summary.update(self._evaluate_dataset(labelled))

        self.writer.write_summary(summary)
        self.writer.write_run_manifest(self._run_manifest(len(records)))
        self.writer.write_report(self.config.experiment_name, summary)
        return EvaluationResult(
            output_dir=self.config.output_dir,
            scene_count=len(records),
            summary=summary,
        )

    def _evaluate_scene(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for metric in self.scene_metrics:
            if ground_truth is None and getattr(metric, "requires_ground_truth", True):
                continue
            value = float(metric.compute(prediction, ground_truth))
            if not math.isnan(value):
                values[metric.name] = value
        return values

    def _evaluate_dataset(
        self, labelled: list[LabelledPrediction]
    ) -> dict[str, dict[str, Any]]:
        predictions = [prediction for prediction, _ in labelled]
        ground_truths = [ground_truth for _, ground_truth in labelled]
        return {
            metric.name: {
                "count": len(labelled),
                "value": float(metric.compute_dataset(predictions, ground_truths)),
                "dataset_level": True,
            }
            for metric in self.dataset_metrics
        }

    def _run_manifest(self, scene_count: int) -> dict[str, Any]:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_revision": git_revision(),
            "seed": self.config.seed,
            "scene_count": scene_count,
            "config": asdict(self.config),
            "estimator": self.prediction_source.provenance(),
        }


def _require_unique_names(metrics: list[Metric]) -> None:
    names = [metric.name for metric in metrics]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Metric names must be unique; give the repeated entries a 'name' "
            f"parameter. Duplicates: {', '.join(duplicates)}"
        )
