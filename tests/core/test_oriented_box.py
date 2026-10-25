import math

import pytest
from typing_extensions import TypeAlias

from pano3dbench.domain import (
    BoundingBox3D,
    OrientedBox3D,
    SceneObject,
    SceneOutput,
    parse_box,
)
from pano3dbench.metrics import ObjectDetectionMAP

IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

Matrix3x3: TypeAlias = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _yaw_basis(angle: float) -> Matrix3x3:
    cos, sin = math.cos(angle), math.sin(angle)
    return (
        (cos, sin, 0.0),
        (-sin, cos, 0.0),
        (0.0, 0.0, 1.0),
    )


def test_identity_basis_matches_axis_aligned_iou() -> None:
    left = BoundingBox3D((0, 0, 0), (2, 2, 2)).as_oriented()
    right = BoundingBox3D((1, 1, 1), (3, 3, 3)).as_oriented()
    assert left.iou(right) == pytest.approx(1 / 15, abs=1e-9)


def test_self_iou_is_one() -> None:
    box = OrientedBox3D((1, 2, 3), (2, 4, 6), _yaw_basis(0.37))
    assert box.iou(box) == pytest.approx(1.0, abs=1e-9)


def test_quarter_turn_of_square_footprint_is_identical() -> None:
    upright = OrientedBox3D((0, 0, 0), (2, 2, 4), IDENTITY)
    turned = OrientedBox3D((0, 0, 0), (2, 2, 4), _yaw_basis(math.pi / 2))
    assert upright.iou(turned) == pytest.approx(1.0, abs=1e-6)


def test_concentric_45_degree_rotation_exact_value() -> None:
    upright = OrientedBox3D((0, 0, 0), (2, 2, 2), IDENTITY)
    turned = OrientedBox3D((0, 0, 0), (2, 2, 2), _yaw_basis(math.pi / 4))
    assert upright.iou(turned) == pytest.approx(1 / math.sqrt(2), abs=1e-5)


def test_no_vertical_overlap_gives_zero() -> None:
    low = OrientedBox3D((0, 0, 0), (2, 2, 2), IDENTITY)
    high = OrientedBox3D((0, 0, 10), (2, 2, 2), IDENTITY)
    assert low.iou(high) == 0.0


def test_disjoint_footprints_give_zero() -> None:
    here = OrientedBox3D((0, 0, 0), (2, 2, 2), IDENTITY)
    far = OrientedBox3D((10, 0, 0), (2, 2, 2), IDENTITY)
    assert here.iou(far) == 0.0


def test_cross_type_iou_is_symmetric() -> None:
    aabb = BoundingBox3D((0, 0, 0), (2, 2, 2))
    oriented = OrientedBox3D((1, 1, 1), (2, 2, 2), _yaw_basis(math.pi / 6))
    assert aabb.iou(oriented) == pytest.approx(oriented.iou(aabb), abs=1e-9)
    assert aabb.iou(oriented) > 0.0


def test_iou_stays_within_unit_interval() -> None:
    a = OrientedBox3D((0, 0, 0), (3, 1, 2), _yaw_basis(0.9))
    b = OrientedBox3D((0.5, 0.2, 0.1), (1, 3, 2), _yaw_basis(-0.4))
    assert 0.0 <= a.iou(b) <= 1.0


def test_dict_round_trip() -> None:
    box = OrientedBox3D((1, 2, 3), (2, 4, 6), _yaw_basis(0.5))
    assert OrientedBox3D.from_dict(box.to_dict()) == box


def test_parse_box_dispatches_on_keys() -> None:
    oriented = parse_box({"center": [0, 0, 0], "size": [1, 1, 1], "basis": IDENTITY})
    axis_aligned = parse_box({"min_corner": [0, 0, 0], "max_corner": [1, 1, 1]})
    assert isinstance(oriented, OrientedBox3D)
    assert isinstance(axis_aligned, BoundingBox3D)


def test_degenerate_size_becomes_zero_volume() -> None:
    box = OrientedBox3D((0, 0, 0), (-1, 1, 1), IDENTITY)
    assert box.size[0] == 0.0
    assert box.volume == 0.0
    assert box.iou(OrientedBox3D((0, 0, 0), (2, 2, 2), IDENTITY)) == 0.0


def test_non_finite_values_are_sanitized() -> None:
    nan = float("nan")
    box = OrientedBox3D((nan, 0, 0), (nan, 2, 2), IDENTITY)
    assert box.center[0] == 0.0
    assert box.size[0] == 0.0
    assert box.volume == 0.0


@pytest.mark.parametrize(("threshold", "expected_ap"), [(0.15, 1.0), (0.8, 0.0)])
def test_object_map_uses_oriented_iou_at_protocol_threshold(
    threshold: float, expected_ap: float
) -> None:
    def scene(box: OrientedBox3D) -> SceneOutput:
        return SceneOutput(
            layout=None, objects=(SceneObject("o", "chair", box, score=0.9),)
        )

    ground_truth = scene(OrientedBox3D((0, 0, 0), (2, 2, 2), IDENTITY))
    prediction = scene(OrientedBox3D((0, 0, 0), (2, 2, 2), _yaw_basis(math.pi / 4)))

    metric = ObjectDetectionMAP({"iou_threshold": threshold})

    assert metric.compute(prediction, ground_truth) == expected_ap
