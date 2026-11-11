from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar



def _grid(start: float, stop: float, count: int) -> tuple[float, ...]:
    step = (stop - start) / (count - 1)
    return (*(start + index * step for index in range(count - 1)), stop)


IOU_THRESHOLDS = _grid(0.5, 0.95, 10)
RECALL_POINTS = _grid(0.0, 1.0, 101)
MAX_DETECTIONS = 100

Region = TypeVar("Region")


@dataclass(frozen=True)
class Labelled(Generic[Region]):
    label: str
    region: Region
    score: float = 1.0
    on_seam: bool = False


@dataclass
class _ClassRecord:
    ground_truth: int = 0
    detections: list[tuple[float, int, tuple[bool, ...]]] = field(default_factory=list)


@dataclass
class _SeamRecord:
    total: int = 0
    found: int = 0


class AveragePrecision(Generic[Region]):
    def __init__(self, iou: Callable[[Region, Region], float]) -> None:
        self._iou = iou
        self._classes: dict[str, _ClassRecord] = defaultdict(_ClassRecord)
        self._seam: dict[bool, _SeamRecord] = defaultdict(_SeamRecord)
        self._images = 0

    def add_image(
        self,
        predictions: Sequence[Labelled[Region]],
        ground_truth: Sequence[Labelled[Region]],
    ) -> None:
        image = self._images
        self._images += 1
        labels = {item.label for item in (*predictions, *ground_truth)}
        for label in sorted(labels):
            truths = [item for item in ground_truth if item.label == label]
            ranked = sorted(
                (item for item in predictions if item.label == label),
                key=lambda item: -item.score,
            )[:MAX_DETECTIONS]
            record = self._classes[label]
            record.ground_truth += len(truths)
            ious = [[self._iou(item.region, truth.region) for truth in truths] for item in ranked]
            matches = [_match(ious, len(truths), threshold) for threshold in IOU_THRESHOLDS]
            for index, item in enumerate(ranked):
                record.detections.append(
                    (item.score, image, tuple(match[0][index] >= 0 for match in matches))
                )
            matched_at_half = matches[0][1]
            for truth_index, truth in enumerate(truths):
                seam = self._seam[truth.on_seam]
                seam.total += 1
                seam.found += matched_at_half[truth_index] >= 0

    def summary(self) -> dict[str, Any]:
        classes = {
            label: _class_summary(record)
            for label, record in sorted(self._classes.items())
            if record.ground_truth
        }
        overall = {
            key: _mean([values[key] for values in classes.values()])
            for key in ("ap", "ap50", "ap75", "recall50", "recall")
        }
        return {
            **overall,
            "images": self._images,
            "gt_objects": sum(values["gt_objects"] for values in classes.values()),
            "predictions": sum(
                len(record.detections) for record in self._classes.values()
            ),
            "classes": classes,
            "seam": {
                "gt_objects": self._seam[True].total,
                "recall50": _ratio(self._seam[True]),
                "rest_gt_objects": self._seam[False].total,
                "rest_recall50": _ratio(self._seam[False]),
            },
        }


def _match(
    ious: list[list[float]], truth_count: int, threshold: float
) -> tuple[list[int], list[int]]:
    taken = [-1] * truth_count
    found = [-1] * len(ious)
    for index, row in enumerate(ious):
        best = min(threshold, 1 - 1e-10)
        chosen = -1
        for truth_index, value in enumerate(row):
            if taken[truth_index] >= 0 or value < best:
                continue
            best = value
            chosen = truth_index
        if chosen >= 0:
            taken[chosen] = index
            found[index] = chosen
    return found, taken


def _class_summary(record: _ClassRecord) -> dict[str, Any]:
    ordered = sorted(
        enumerate(record.detections), key=lambda item: (-item[1][0], item[1][1], item[0])
    )
    precisions: list[float] = []
    recalls: list[float] = []
    for threshold_index in range(len(IOU_THRESHOLDS)):
        precision, recall = _precision_at_recall_points(
            [entry[2][threshold_index] for _, entry in ordered], record.ground_truth
        )
        precisions.append(precision)
        recalls.append(recall)
    return {
        "ap": _mean(precisions),
        "ap50": precisions[0],
        "ap75": precisions[5],
        "recall50": recalls[0],
        "recall": _mean(recalls),
        "gt_objects": record.ground_truth,
        "predictions": len(record.detections),
    }


def _precision_at_recall_points(
    hits: Sequence[bool], ground_truth: int
) -> tuple[float, float]:
    true_positives = false_positives = 0
    precision: list[float] = []
    recall: list[float] = []
    for hit in hits:
        true_positives += hit
        false_positives += not hit
        precision.append(
            true_positives / (true_positives + false_positives + 2.220446049250313e-16)
        )
        recall.append(true_positives / ground_truth)
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    sampled = 0.0
    position = 0
    for point in RECALL_POINTS:
        while position < len(recall) and recall[position] < point:
            position += 1
        if position < len(recall):
            sampled += precision[position]
    return sampled / len(RECALL_POINTS), (recall[-1] if recall else 0.0)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(record: _SeamRecord) -> float:
    return record.found / record.total if record.total else 0.0
