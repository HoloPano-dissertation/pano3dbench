from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pano3dbench.file_formats.mesh_files import read_obj, read_ply
from pano3dbench.file_formats.point_files import as_points, read_f32_points
from pano3dbench.geometry.mesh_sampling import sample_surface
from pano3dbench.geometry.point_cloud import Point, chamfer_and_fscore, normalize_pair
from pano3dbench.metrics.base import METRICS, Metric
from pano3dbench.metrics.matching import greedy_match
from pano3dbench.scene import SceneObject, SceneOutput


def load_shape_points(
    spec: Any, *, num_points: int, rng, mesh_root: Path | None = None
) -> list[Point] | None:
    if spec is None:
        return None
    if isinstance(spec, dict):
        if "points" in spec:
            return as_points(spec["points"])
        spec = spec.get("file") or spec.get("mesh") or spec.get("path")
    if isinstance(spec, (list, tuple)):
        return as_points(spec)
    if not isinstance(spec, str):
        raise ValueError(f"unsupported shape reference: {type(spec).__name__}")

    path = Path(spec).expanduser()
    if mesh_root is not None and not path.is_absolute():
        path = Path(mesh_root) / path
    suffix = path.suffix.lower()
    if suffix == ".json":
        return as_points(json.loads(path.read_text(encoding="utf-8")))
    if suffix in (".f32", ".bin"):
        return read_f32_points(path)
    if suffix == ".ply":
        return sample_surface(*read_ply(path), num_points, rng)
    if suffix == ".obj":
        return sample_surface(*read_obj(path), num_points, rng)
    raise ValueError(f"unsupported shape file '{path.name}'")


def _match_pairs(
    predictions: Sequence[SceneObject],
    labels: Sequence[SceneObject],
    iou_threshold: float,
) -> list[tuple[SceneObject, SceneObject]]:
    return [
        (prediction, label)
        for prediction, label in greedy_match(predictions, labels, iou_threshold)
        if label is not None
    ]


class _ShapeMetric(Metric):
    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        parameters = parameters or {}
        self.iou_threshold = float(parameters.get("iou_threshold", 0.15))
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in the interval (0, 1]")
        self.num_points = int(parameters.get("num_points", 2048))
        if self.num_points <= 0:
            raise ValueError("num_points must be positive")
        self.seed = int(parameters.get("seed", 0))
        self.squared = bool(parameters.get("squared", True))
        self.fscore_threshold = float(parameters.get("fscore_threshold", 0.1))
        self.normalize = str(parameters.get("normalize", "none"))
        mesh_root = parameters.get("mesh_root")
        self.mesh_root = Path(mesh_root) if mesh_root else None

    def _pair_points(self, obj: SceneObject):
        return load_shape_points(
            obj.attributes.get("shape"),
            num_points=self.num_points,
            rng=random.Random(self.seed),
            mesh_root=self.mesh_root,
        )

    def _reduce_pair(self, pred_points, gt_points) -> float:
        raise NotImplementedError

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        if ground_truth is None:
            raise ValueError(f"{self.name} requires ground truth")
        values: list[float] = []
        for pred_obj, gt_obj in _match_pairs(
            prediction.objects, ground_truth.objects, self.iou_threshold
        ):
            pred_points = self._pair_points(pred_obj)
            gt_points = self._pair_points(gt_obj)
            if not pred_points or not gt_points:
                continue
            pred_points, gt_points = normalize_pair(
                pred_points, gt_points, self.normalize
            )
            value = self._reduce_pair(pred_points, gt_points)
            if not math.isnan(value):
                values.append(value)
        if not values:
            return math.nan
        return sum(values) / len(values)


@METRICS.register("mesh_chamfer")
class MeshChamfer(_ShapeMetric):
    name = "mesh_chamfer"

    def _reduce_pair(self, pred_points, gt_points) -> float:
        chamfer, _ = chamfer_and_fscore(
            pred_points,
            gt_points,
            squared=self.squared,
            fscore_threshold=self.fscore_threshold,
        )
        return chamfer


@METRICS.register("mesh_fscore")
class MeshFScore(_ShapeMetric):
    name = "mesh_fscore"

    def _reduce_pair(self, pred_points, gt_points) -> float:
        _, fscore = chamfer_and_fscore(
            pred_points,
            gt_points,
            squared=self.squared,
            fscore_threshold=self.fscore_threshold,
        )
        return fscore
