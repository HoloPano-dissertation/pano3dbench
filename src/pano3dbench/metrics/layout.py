from __future__ import annotations

from typing import Any

from pano3dbench.metrics.base import METRICS, Metric
from pano3dbench.scene import SceneOutput


@METRICS.register("layout_iou_3d")
class LayoutIoU3D(Metric):
    name = "layout_iou_3d"

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        del parameters

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        if ground_truth is None:
            raise ValueError("Layout 3D IoU requires ground truth")
        if ground_truth.layout is None:
            return float("nan")
        if prediction.layout is None:
            return 0.0
        return prediction.layout.iou(ground_truth.layout)
