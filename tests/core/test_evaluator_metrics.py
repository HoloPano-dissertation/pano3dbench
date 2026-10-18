import math
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import pytest

from benchmark_toolbox.config import ComponentConfig, ExperimentConfig
from benchmark_toolbox.datasets.base import BaseDatasetLoader
from benchmark_toolbox.domain import (
    BoundingBox3D,
    SceneObject,
    SceneOutput,
    SceneSample,
)
from benchmark_toolbox.evaluator import Evaluator, _aggregate
from benchmark_toolbox.metrics.base import BaseMetric
from benchmark_toolbox.metrics import (
    CollisionRate,
    LayoutIoU3D,
    ObjectDetectionMAPDataset,
)
from benchmark_toolbox.models.base import BaseSceneEstimator

UNIT_BOX = BoundingBox3D((0, 0, 0), (1, 1, 1))


class _NaNWhenEmptyMetric(BaseMetric):
    name = "nan_when_empty"
    requires_ground_truth = False

    def compute(
        self,
        prediction: SceneOutput,
        ground_truth: Optional[SceneOutput],  # noqa: ARG002
    ) -> float:
        return 1.0 if prediction.objects else math.nan


class _ListDataset(BaseDatasetLoader):
    def __init__(self, samples: list[SceneSample]) -> None:
        self._samples = samples

    def samples(self) -> Iterator[SceneSample]:
        return iter(self._samples)


class _PerSampleEstimator(BaseSceneEstimator):
    def __init__(self, outputs: dict[str, SceneOutput]) -> None:
        self._outputs = outputs

    def predict(self, sample: SceneSample) -> SceneOutput:
        return self._outputs[sample.sample_id]


class _FixedEstimator(BaseSceneEstimator):
    def __init__(self, output: SceneOutput) -> None:
        self._output = output

    def predict(self, sample: SceneSample) -> SceneOutput:  # noqa: ARG002
        return self._output


def _config(output_dir: Path) -> ExperimentConfig:
    dummy = ComponentConfig(type="x", parameters={})
    return ExperimentConfig(
        experiment_name="t",
        seed=42,
        output_dir=output_dir,
        model=dummy,
        dataset=dummy,
        metrics=(),
        source_path=output_dir / "c.json",
    )


def test_duplicate_metric_names_are_rejected(tmp_path: Path) -> None:
    metrics = [
        ObjectDetectionMAPDataset({"iou_threshold": 0.15, "classes": ["chair"]}),
        ObjectDetectionMAPDataset({"iou_threshold": 0.15, "classes": ["sofa"]}),
    ]
    with pytest.raises(ValueError, match="Metric names must be unique"):
        Evaluator(
            _config(tmp_path),
            _FixedEstimator(SceneOutput(layout=None)),
            _ListDataset([]),
            metrics,
        )


def test_evaluator_skips_metrics_without_ground_truth(tmp_path: Path) -> None:
    prediction = SceneOutput(
        layout=UNIT_BOX,
        objects=(
            SceneObject("a", "chair", UNIT_BOX),
            SceneObject("b", "chair", UNIT_BOX),
        ),
    )
    sample = SceneSample(sample_id="s1", input_path=Path(__file__), ground_truth=None)

    result = Evaluator(
        _config(tmp_path),
        _FixedEstimator(prediction),
        _ListDataset([sample]),
        [LayoutIoU3D(), CollisionRate()],  # ty: ignore[missing-argument]
    ).run()

    assert result.summary["layout_iou_3d"]["count"] == 0
    assert result.summary["collision_rate"]["count"] == 1
    assert result.summary["collision_rate"]["mean"] == 1.0
    assert "mean" not in result.summary["layout_iou_3d"]
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| layout_iou_3d | 0 | n/a | n/a | n/a |" in report


def test_a_sample_id_with_folders_gets_its_prediction_copy(tmp_path: Path) -> None:
    prediction = SceneOutput(layout=UNIT_BOX, objects=())
    sample = SceneSample(
        sample_id="house-1/Bedroom-2/3", input_path=Path(__file__), ground_truth=None
    )

    Evaluator(
        _config(tmp_path),
        _FixedEstimator(prediction),
        _ListDataset([sample]),
        [CollisionRate()],
    ).run()

    assert (tmp_path / "predictions" / "house-1" / "Bedroom-2" / "3.json").is_file()


def test_layout_iou_is_not_applicable_without_ground_truth_layout() -> None:
    boxed = SceneOutput(layout=UNIT_BOX)
    assert math.isnan(LayoutIoU3D().compute(boxed, SceneOutput(layout=None)))  # ty: ignore[missing-argument]
    with pytest.raises(ValueError, match="Layout 3D IoU requires ground truth"):
        LayoutIoU3D().compute(boxed, None)  # ty: ignore[missing-argument]


def test_evaluator_excludes_nan_from_aggregation(tmp_path: Path) -> None:
    with_object = SceneOutput(
        layout=None, objects=(SceneObject("a", "chair", UNIT_BOX),)
    )
    empty = SceneOutput(layout=None, objects=())
    samples = [
        SceneSample("s1", Path(__file__), ground_truth=with_object),
        SceneSample("s2", Path(__file__), ground_truth=empty),
    ]
    result = Evaluator(
        _config(tmp_path),
        _PerSampleEstimator({"s1": with_object, "s2": empty}),
        _ListDataset(samples),
        [_NaNWhenEmptyMetric()],
    ).run()

    assert result.summary["nan_when_empty"]["count"] == 1
    assert result.summary["nan_when_empty"]["mean"] == 1.0


def test_dataset_map_perfect_predictions() -> None:
    scene = SceneOutput(
        layout=None,
        objects=(
            SceneObject("c", "chair", BoundingBox3D((0, 0, 0), (1, 1, 1)), 0.9),
            SceneObject("t", "table", BoundingBox3D((2, 0, 0), (3, 1, 1)), 0.8),
        ),
    )
    metric = ObjectDetectionMAPDataset({"iou_threshold": 0.5})
    assert metric.compute_dataset([scene, scene], [scene, scene]) == 1.0  # ty: ignore[unresolved-attribute]


def test_aggregate_has_bootstrap_ci() -> None:
    aggregate = _aggregate([0.5, 0.5, 0.5, 0.5], seed=42)
    assert aggregate["count"] == 4
    assert aggregate["mean"] == pytest.approx(0.5)
    assert aggregate["ci95_low"] == pytest.approx(0.5)
    assert aggregate["ci95_high"] == pytest.approx(0.5)


def test_aggregate_single_value() -> None:
    aggregate = _aggregate([0.7], seed=1)
    assert aggregate["ci95_low"] == aggregate["ci95_high"]
