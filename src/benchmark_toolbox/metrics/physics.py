from __future__ import annotations

from typing import Any

from benchmark_toolbox.domain import SceneOutput
from benchmark_toolbox.metrics.base import BaseMetric


@BaseMetric.registry.register("collision_rate")
class CollisionRate(BaseMetric):
    name = "collision_rate"
    requires_ground_truth = False

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        parameters = parameters or {}
        self.minimum_intersection = float(
            parameters.get("minimum_intersection", 0.0)
        )

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        del ground_truth
        pair_count = 0
        collision_count = 0
        for left_index, left in enumerate(prediction.objects):
            for right in prediction.objects[left_index + 1 :]:
                pair_count += 1
                if (
                    left.bbox.intersection_volume(right.bbox)
                    > self.minimum_intersection
                ):
                    collision_count += 1
        return collision_count / pair_count if pair_count else 0.0


@BaseMetric.registry.register("layout_violation_rate")
class LayoutViolationRate(BaseMetric):
    name = "layout_violation_rate"
    requires_ground_truth = False

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        parameters = parameters or {}
        self.tolerance = float(parameters.get("tolerance", 0.0))
        if not 0.0 <= self.tolerance < 1.0:
            raise ValueError("tolerance must be in the interval [0, 1)")

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        del ground_truth
        layout = prediction.layout
        if layout is None:
            return float("nan")
        if not prediction.objects:
            return 0.0
        violations = 0
        considered = 0
        for scene_object in prediction.objects:
            volume = scene_object.bbox.volume
            if volume <= 0.0:
                continue
            considered += 1
            inside = scene_object.bbox.intersection_volume(layout)
            outside_fraction = 1.0 - inside / volume
            if outside_fraction > self.tolerance + 1e-9:
                violations += 1
        return violations / considered if considered else 0.0


@BaseMetric.registry.register("layout_penetration")
class LayoutPenetration(BaseMetric):
    name = "layout_penetration"
    requires_ground_truth = False

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        del parameters

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        del ground_truth
        layout = prediction.layout
        if layout is None:
            return float("nan")
        if not prediction.objects:
            return 0.0
        total = 0.0
        considered = 0
        for scene_object in prediction.objects:
            volume = scene_object.bbox.volume
            if volume <= 0.0:
                continue
            considered += 1
            inside = scene_object.bbox.intersection_volume(layout)
            total += max(0.0, 1.0 - inside / volume)
        return total / considered if considered else 0.0
