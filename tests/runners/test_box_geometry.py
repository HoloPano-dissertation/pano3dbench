import math

import pytest
import scene_payload as payload

IDENTITY = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]


def test_identity_box() -> None:
    box = payload.oriented_box_to_axis_aligned([0, 0, 0], IDENTITY, [1, 2, 3])
    assert box["min_corner"] == [-1.0, -2.0, -3.0]
    assert box["max_corner"] == [1.0, 2.0, 3.0]


def test_rotation_45_about_z_grows_footprint() -> None:
    cos = math.cos(math.pi / 4)
    sin = math.sin(math.pi / 4)
    rotate_about_z = [
        [cos, sin, 0],
        [-sin, cos, 0],
        [0, 0, 1],
    ]
    box = payload.oriented_box_to_axis_aligned([0, 0, 0], rotate_about_z, [1, 1, 5])
    assert box["max_corner"] == pytest.approx(
        [math.sqrt(2), math.sqrt(2), 5.0], abs=1e-5
    )


def test_corners_to_axis_aligned() -> None:
    box = payload.corners_to_axis_aligned([[0, 0, 0], [1, 2, 3], [-1, 0, 1]])
    assert box["min_corner"] == [-1.0, 0.0, 0.0]
    assert box["max_corner"] == [1.0, 2.0, 3.0]


def test_scene_from_oriented_objects() -> None:
    objects = [
        {
            "label": "chair",
            "centroid": [0, 0, 0],
            "basis": IDENTITY,
            "coeffs": [1, 1, 1],
        }
    ]
    scene = payload.scene_from_oriented_objects(
        objects, layout_corners=[[0, 0, 0], [4, 3, 5]]
    )

    assert len(scene["objects"]) == 1
    assert scene["objects"][0]["label"] == "chair"
    assert scene["layout"]["max_corner"] == [4.0, 3.0, 5.0]
