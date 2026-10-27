import math
from typing import Union

import pytest

from pano3dbench.geometry.boxes import BoundingBox3D, OrientedBox3D
from pano3dbench.scene import SceneObject, SceneOutput
from pano3dbench.metrics import LayoutPenetration, LayoutViolationRate

IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
ROOM = BoundingBox3D((0, 0, 0), (10, 10, 3))

INSIDE = BoundingBox3D((1, 1, 0), (2, 2, 1))
THROUGH_WALL = BoundingBox3D((9, 9, 0), (12, 11, 1))
THROUGH_CEILING = BoundingBox3D((4, 4, 0), (6, 6, 4))


def _scene(*objects: SceneObject) -> SceneOutput:
    return SceneOutput(layout=ROOM, objects=tuple(objects))


def _obj(object_id: str, box: Union[BoundingBox3D, OrientedBox3D]) -> SceneObject:
    return SceneObject(object_id, "thing", box)


def _no_layout_scene() -> SceneOutput:
    return SceneOutput(
        layout=None, objects=(_obj("a", BoundingBox3D((0, 0, 0), (1, 1, 1))),)
    )


@pytest.mark.parametrize(
    ("box", "expected"),
    [(INSIDE, 0.0), (THROUGH_WALL, 1.0), (THROUGH_CEILING, 1.0)],
    ids=["inside", "through-wall", "through-ceiling"],
)
def test_violation_rate_single_object(box: BoundingBox3D, expected: float) -> None:
    assert LayoutViolationRate().compute(_scene(_obj("a", box)), None) == expected  # ty: ignore[missing-argument]


def test_mixed_objects_give_fraction() -> None:
    scene = _scene(_obj("inside", INSIDE), _obj("outside", THROUGH_WALL))
    assert LayoutViolationRate().compute(scene, None) == 0.5  # ty: ignore[missing-argument]


@pytest.mark.parametrize(("tolerance", "expected"), [(0.3, 0.0), (0.0, 1.0)])
def test_tolerance_forgives_small_protrusion(tolerance: float, expected: float) -> None:
    scene = _scene(_obj("a", THROUGH_CEILING))
    assert (
        LayoutViolationRate({"tolerance": tolerance}).compute(scene, None) == expected
    )


def test_violation_rate_without_layout_is_not_applicable() -> None:
    assert math.isnan(LayoutViolationRate().compute(_no_layout_scene(), None))  # ty: ignore[missing-argument]


def test_oriented_object_outside_room_is_a_violation() -> None:
    box = OrientedBox3D((9.5, 9.5, 1.5), (2, 2, 2), IDENTITY)
    assert LayoutViolationRate().compute(_scene(_obj("a", box)), None) == 1.0  # ty: ignore[missing-argument]


def test_rejects_out_of_range_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance must be in the interval"):
        LayoutViolationRate({"tolerance": 1.0})


def test_object_inside_has_zero_penetration() -> None:
    assert LayoutPenetration().compute(_scene(_obj("a", INSIDE)), None) == 0.0  # ty: ignore[missing-argument]


def test_penetration_is_outside_volume_fraction() -> None:
    value = LayoutPenetration().compute(_scene(_obj("a", THROUGH_CEILING)), None)  # ty: ignore[missing-argument]
    assert value == pytest.approx(0.25, abs=1e-6)


def test_penetration_averages_over_objects() -> None:
    scene = _scene(_obj("inside", INSIDE), _obj("quarter", THROUGH_CEILING))
    assert LayoutPenetration().compute(scene, None) == pytest.approx(0.125, abs=1e-6)  # ty: ignore[missing-argument]


def test_penetration_without_layout_is_not_applicable() -> None:
    assert math.isnan(LayoutPenetration().compute(_no_layout_scene(), None))  # ty: ignore[missing-argument]
