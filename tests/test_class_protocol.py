import math

import pytest
from typing_extensions import TypeAlias

from benchmark_toolbox.domain import BoundingBox3D, SceneObject, SceneOutput
from benchmark_toolbox.metrics.geometry import (
    ObjectDetectionMAP,
    ObjectDetectionMAPDataset,
)

UNIT = ((0, 0, 0), (1, 1, 1))
FAR = ((10, 0, 0), (11, 1, 1))

Corner: TypeAlias = tuple[float, float, float]


def scene_object(
    object_id: str, label: str, minimum: Corner, maximum: Corner, score: float = 1.0
) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        label=label,
        score=score,
        bbox=BoundingBox3D(minimum, maximum),
    )


def scene(*objects: SceneObject) -> SceneOutput:
    return SceneOutput(layout=None, objects=objects)


def test_classes_restrict_the_average_to_the_protocol() -> None:
    ground_truth = scene(
        scene_object("a", "chair", *UNIT), scene_object("b", "plant", *FAR)
    )
    prediction = scene(scene_object("pa", "chair", *UNIT, 0.9))

    unrestricted = ObjectDetectionMAP({"iou_threshold": 0.15}).compute(
        prediction, ground_truth
    )
    restricted = ObjectDetectionMAP(
        {"iou_threshold": 0.15, "classes": ["chair"]}
    ).compute(prediction, ground_truth)

    assert unrestricted == pytest.approx(0.5)
    assert restricted == pytest.approx(1.0)


def test_out_of_protocol_prediction_is_not_a_false_positive() -> None:
    ground_truth = scene(scene_object("a", "chair", *UNIT))
    prediction = scene(
        scene_object("junk", "plant", (50, 50, 50), (51, 51, 51), 0.99),
        scene_object("pa", "chair", *UNIT, 0.5),
    )
    value = ObjectDetectionMAP({"iou_threshold": 0.15, "classes": ["chair"]}).compute(
        prediction, ground_truth
    )
    assert value == pytest.approx(1.0)


def test_class_map_merges_source_labels() -> None:
    ground_truth = scene(
        scene_object("a", "office_chair", *UNIT), scene_object("b", "stool", *FAR)
    )
    prediction = scene(
        scene_object("pa", "office_chair", *UNIT, 0.9),
        scene_object("pb", "stool", *FAR, 0.8),
    )
    metric = ObjectDetectionMAP(
        {
            "iou_threshold": 0.15,
            "class_map": {"office_chair": "chair", "stool": "chair"},
        }
    )
    assert metric.compute(prediction, ground_truth) == pytest.approx(1.0)


def test_class_map_drops_unmapped_labels() -> None:
    ground_truth = scene(
        scene_object("a", "office_chair", *UNIT), scene_object("b", "plant", *FAR)
    )
    prediction = scene(scene_object("pa", "office_chair", *UNIT, 0.9))

    value = ObjectDetectionMAP(
        {"iou_threshold": 0.15, "class_map": {"office_chair": "chair"}}
    ).compute(prediction, ground_truth)

    assert value == pytest.approx(1.0)


def test_scene_without_protocol_classes_is_not_applicable() -> None:
    ground_truth = scene(scene_object("b", "plant", *UNIT))
    value = ObjectDetectionMAP({"iou_threshold": 0.15, "classes": ["chair"]}).compute(
        scene(), ground_truth
    )
    assert math.isnan(value)


def test_without_protocol_empty_ground_truth_keeps_legacy_zero() -> None:
    value = ObjectDetectionMAP({"iou_threshold": 0.15}).compute(scene(), scene())
    assert value == 0.0


# EXPLANATION: Wrong type hints of decorator confuse type checker
# TODO: Remove suppresions after fix
def test_dataset_metric_restricts_and_renames() -> None:
    ground_truth = scene(
        scene_object("a", "office_chair", *UNIT), scene_object("b", "plant", *FAR)
    )
    prediction = scene(scene_object("pa", "office_chair", *UNIT, 0.9))
    metric = ObjectDetectionMAPDataset(
        {
            "iou_threshold": 0.15,
            "class_map": {"office_chair": "chair"},
            "classes": ["chair"],
            "name": "ap_chair",
        }
    )

    assert metric.compute_dataset([prediction], [ground_truth]) == pytest.approx(1.0)  # ty: ignore[unresolved-attribute]
    assert metric.name == "ap_chair"


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (ObjectDetectionMAP, "object_map"),
        (ObjectDetectionMAPDataset, "object_map_dataset"),
    ],
)
def test_name_defaults_to_the_metric_type(cls, expected: str) -> None:  # noqa: ANN001
    assert cls({"iou_threshold": 0.15}).name == expected


@pytest.mark.parametrize("parameters", [{"classes": "chair"}, {"class_map": ["chair"]}])
def test_invalid_protocol_parameters_are_rejected(parameters: dict) -> None:
    with pytest.raises(ValueError, match="class"):
        ObjectDetectionMAP(parameters)
