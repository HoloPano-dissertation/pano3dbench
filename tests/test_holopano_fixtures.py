import math
from pathlib import Path

import pytest
from _loading import load_module
from typing_extensions import TypeAlias

from benchmark_toolbox.domain import OrientedBox3D

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "holopano_to_fixtures.py"
holopano_to_fixtures = load_module("holopano_to_fixtures", _SCRIPT)

Point3D: TypeAlias = tuple[float, float, float]
Radians: TypeAlias = float


def reference_corners(
    centroid: Point3D, size: tuple[float, float, float], yaw: Radians
) -> list[tuple[float, ...]]:
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    corners = []
    for sign_x in (-0.5, 0.5):
        for sign_y in (-0.5, 0.5):
            for sign_z in (-0.5, 0.5):
                local = (sign_x * size[0], sign_y * size[1], sign_z * size[2])
                corners.append(
                    (
                        centroid[0] + cos_yaw * local[0] - sin_yaw * local[1],
                        centroid[1] + sin_yaw * local[0] + cos_yaw * local[1],
                        centroid[2] + local[2],
                    )
                )
    return sorted(tuple(round(value, 9) for value in corner) for corner in corners)


def make_box(
    centroid: Point3D, size: tuple[float, float, float], yaw: Radians
) -> OrientedBox3D:
    return OrientedBox3D.from_dict(
        holopano_to_fixtures._oriented_box(centroid, size, yaw)
    )


@pytest.mark.parametrize(
    ("centroid", "size", "yaw"),
    [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0),
        ((1.5, -2.0, 0.8), (0.6, 1.2, 0.9), math.pi / 2),
        ((-3.0, 0.25, 1.1), (2.0, 0.5, 1.7), 0.7),
        ((0.4, 0.4, 0.4), (1.0, 3.0, 0.2), -2.3),
    ],
    ids=lambda value: f"{value:.2f}" if isinstance(value, float) else None,
)
def test_basis_matches_holopano_corner_convention(
    centroid: Point3D, size: tuple[float, float, float], yaw: Radians
) -> None:
    produced = sorted(
        tuple(round(value, 9) for value in corner)
        for corner in make_box(centroid, size, yaw).corners()
    )
    expected = reference_corners(centroid, size, yaw)

    assert len(produced) == len(expected)
    for produced_corner, expected_corner in zip(produced, expected):
        assert produced_corner == pytest.approx(expected_corner, abs=1e-7)


def test_identical_boxes_have_iou_one_and_rotated_ones_do_not() -> None:
    size = (2.0, 1.0, 1.0)
    first = make_box((0.0, 0.0, 0.0), size, 0.0)
    same = make_box((0.0, 0.0, 0.0), size, 0.0)
    rotated = make_box((0.0, 0.0, 0.0), size, math.pi / 2)

    assert first.iou(same) == pytest.approx(1.0)
    assert first.iou(rotated) == pytest.approx(1.0 / 3.0, abs=1e-6)
