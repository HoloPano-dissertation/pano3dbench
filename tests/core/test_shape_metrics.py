import math
import random
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import pytest
from typing_extensions import TypeAlias

from pano3dbench.domain import BoundingBox3D, SceneObject, SceneOutput
from pano3dbench.metrics.base import BaseMetric
from pano3dbench.metrics.shape import (
    MeshChamfer,
    MeshFScore,
    _build_kd,
    _nearest_sq,
    chamfer_and_fscore,
    load_shape_points,
    write_f32_points,
)

ORIGIN = (0.0, 0.0, 0.0)

Point3D: TypeAlias = tuple[float, float, float]


def _nearest_sq_brute(points: Iterable[Point3D], query: Point3D) -> float:
    qx, qy, qz = query
    best = math.inf
    for px, py, pz in points:
        dx, dy, dz = qx - px, qy - py, qz - pz
        best = min(best, dx * dx + dy * dy + dz * dz)
    return best


def _shape_object(
    object_id: str,
    label: str,
    minimum: Point3D,
    maximum: Point3D,
    shape: Iterable[Point3D],
    score: float = 1.0,
) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        label=label,
        score=score,
        bbox=BoundingBox3D(minimum, maximum),
        attributes={"shape": shape},
    )


def _scene(*objects: SceneObject) -> SceneOutput:
    return SceneOutput(layout=None, objects=tuple(objects))


def test_kdtree_matches_brute_force() -> None:
    rng = random.Random(7)
    cloud = [
        (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(200)
    ]
    tree = _build_kd(list(cloud))
    for _ in range(80):
        query = (rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-6, 6))
        assert _nearest_sq(tree, query) == pytest.approx(
            _nearest_sq_brute(cloud, query), abs=1e-10
        )


def test_kdtree_finds_exact_member() -> None:
    cloud = [ORIGIN, (1.0, 2.0, 3.0), (-4.0, 5.0, 6.0)]
    tree = _build_kd(list(cloud))
    assert _nearest_sq(tree, (1.0, 2.0, 3.0)) == 0.0


def test_identical_clouds_are_perfect() -> None:
    cloud = [ORIGIN, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    chamfer, fscore = chamfer_and_fscore(cloud, cloud, fscore_threshold=0.01)
    assert chamfer == 0.0
    assert fscore == 1.0


def test_single_point_offset_squared() -> None:
    chamfer, _ = chamfer_and_fscore([(0.3, 0.0, 0.0)], [ORIGIN])
    assert chamfer == pytest.approx(2.0 * 0.3 * 0.3)


def test_unsquared_is_euclidean() -> None:
    chamfer, _ = chamfer_and_fscore([(0.3, 0.0, 0.0)], [ORIGIN], squared=False)
    assert chamfer == pytest.approx(2.0 * 0.3)


@pytest.mark.parametrize(("threshold", "expected"), [(0.5, 1.0), (0.1, 0.0)])
def test_fscore_threshold_gates_match(threshold: float, expected: float) -> None:
    _, fscore = chamfer_and_fscore(
        [(0.3, 0.0, 0.0)], [ORIGIN], fscore_threshold=threshold
    )
    assert fscore == expected


def test_empty_cloud_is_nan() -> None:
    chamfer, fscore = chamfer_and_fscore([], [ORIGIN])
    assert math.isnan(chamfer)
    assert math.isnan(fscore)


QUAD_OBJ = b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n"

QUAD_PLY_ASCII = (
    "ply\nformat ascii 1.0\n"
    "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
    "element face 2\nproperty list uchar int vertex_indices\nend_header\n"
    "0 0 0\n1 0 0\n1 1 0\n0 1 0\n3 0 1 2\n3 0 2 3\n"
).encode("ascii")


def _quad_ply_binary() -> bytes:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
        "element face 2\nproperty list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    body = b"".join(struct.pack("<fff", *v) for v in vertices)
    body += struct.pack("<Biii", 3, 0, 1, 2) + struct.pack("<Biii", 3, 0, 2, 3)
    return header + body


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param(
            [[0, 0, 0], [1, 2, 3]], [ORIGIN, (1.0, 2.0, 3.0)], id="inline-list"
        ),
        pytest.param({"points": [[1, 1, 1]]}, [(1.0, 1.0, 1.0)], id="dict"),
    ],
)
def test_inline_shapes(
    shape: Any,  # noqa: ANN401
    expected: Optional[tuple[float, float, float]],
) -> None:
    assert load_shape_points(shape, num_points=4, rng=random.Random(0)) == expected


def test_f32_roundtrip(tmp_path: Path) -> None:
    cloud = [(0.5, -1.25, 2.0), (3.0, 4.0, 5.0)]
    path = tmp_path / "pts.f32"
    write_f32_points(path, cloud)

    loaded = load_shape_points(str(path), num_points=4, rng=random.Random(0))

    assert loaded is not None
    assert len(loaded) == len(cloud)
    for got, want in zip(loaded, cloud):
        assert got == pytest.approx(want, abs=1e-5)


def test_json_file(tmp_path: Path) -> None:
    path = tmp_path / "pts.json"
    path.write_text("[[0,0,0],[2,2,2]]", encoding="utf-8")
    loaded = load_shape_points(str(path), num_points=4, rng=random.Random(0))
    assert loaded == [ORIGIN, (2.0, 2.0, 2.0)]


@pytest.mark.parametrize(
    ("filename", "content", "count", "seed"),
    [
        pytest.param("quad.obj", QUAD_OBJ, 200, 1, id="obj"),
        pytest.param("quad.ply", QUAD_PLY_ASCII, 100, 2, id="ply-ascii"),
        pytest.param("quad_bin.ply", _quad_ply_binary(), 100, 4, id="ply-binary"),
    ],
)
def test_mesh_sampling_lies_on_the_surface(
    tmp_path: Path, filename: str, content: bytes, count: int, seed: int
) -> None:
    path = tmp_path / filename
    path.write_bytes(content)

    points = load_shape_points(str(path), num_points=count, rng=random.Random(seed))

    assert points is not None
    assert len(points) == count
    for x, y, z in points:
        assert z == pytest.approx(0.0, abs=1e-6)
        assert -1e-6 <= x <= 1 + 1e-6
        assert -1e-6 <= y <= 1 + 1e-6


def test_sampling_is_deterministic_with_seed(tmp_path: Path) -> None:
    path = tmp_path / "quad.obj"
    path.write_bytes(QUAD_OBJ)
    first = load_shape_points(str(path), num_points=50, rng=random.Random(3))
    second = load_shape_points(str(path), num_points=50, rng=random.Random(3))
    assert first == second


def test_registered_in_registry() -> None:
    assert isinstance(BaseMetric.create("mesh_chamfer"), MeshChamfer)  # ty: ignore[invalid-argument-type]
    assert isinstance(BaseMetric.create("mesh_fscore"), MeshFScore)  # ty: ignore[invalid-argument-type]


def test_perfect_shape_is_zero() -> None:
    cloud = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    prediction = _scene(_shape_object("p", "chair", (0, 0, 0), (1, 1, 1), cloud))
    ground_truth = _scene(_shape_object("g", "chair", (0, 0, 0), (1, 1, 1), cloud))

    assert MeshChamfer().compute(prediction, ground_truth) == 0.0  # ty: ignore[missing-argument]
    assert (
        MeshFScore({"fscore_threshold": 0.1}).compute(prediction, ground_truth) == 1.0
    )


def test_averages_only_matched_pairs() -> None:
    prediction = _scene(
        _shape_object("pa", "chair", (0, 0, 0), (1, 1, 1), [(0, 0, 0)]),
        _shape_object("pb", "table", (50, 50, 50), (51, 51, 51), [(9, 9, 9)]),
    )
    ground_truth = _scene(
        _shape_object("ga", "chair", (0, 0, 0), (1, 1, 1), [(0, 0, 0)]),
        _shape_object("gb", "table", (0, 0, 0), (1, 1, 1), [(0, 0, 0)]),
    )
    assert MeshChamfer({"iou_threshold": 0.15}).compute(prediction, ground_truth) == 0.0


def test_no_match_returns_nan() -> None:
    prediction = _scene(_shape_object("p", "chair", (0, 0, 0), (1, 1, 1), [(0, 0, 0)]))
    ground_truth = _scene(
        _shape_object("g", "chair", (90, 90, 90), (91, 91, 91), [(0, 0, 0)])
    )
    assert math.isnan(MeshChamfer().compute(prediction, ground_truth))  # ty: ignore[missing-argument]


def test_pair_without_shape_is_skipped() -> None:
    prediction = _scene(_shape_object("p", "chair", (0, 0, 0), (1, 1, 1), [(0, 0, 0)]))
    ground_truth = _scene(
        SceneObject("g", "chair", BoundingBox3D((0, 0, 0), (1, 1, 1)))
    )
    assert math.isnan(MeshChamfer().compute(prediction, ground_truth))  # ty: ignore[missing-argument]


def test_normalize_unit_scales_by_gt_diagonal() -> None:
    prediction = _scene(
        _shape_object("p", "x", (0, 0, 0), (1, 1, 1), [(1, 0, 0), (3, 2, 2)])
    )
    ground_truth = _scene(
        _shape_object("g", "x", (0, 0, 0), (1, 1, 1), [(0, 0, 0), (2, 2, 2)])
    )

    value = MeshChamfer({"normalize": "unit"}).compute(prediction, ground_truth)

    assert value == pytest.approx(2.0 * (1.0 / math.sqrt(12.0)) ** 2)


def test_normalize_canonical_is_scale_invariant() -> None:
    unit = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    big = [(3 * x + 5, 3 * y + 5, 3 * z + 5) for x, y, z in unit]
    prediction = _scene(_shape_object("p", "x", (0, 0, 0), (1, 1, 1), big))
    ground_truth = _scene(_shape_object("g", "x", (0, 0, 0), (1, 1, 1), unit))

    canonical = MeshChamfer({"normalize": "canonical"}).compute(
        prediction, ground_truth
    )
    raw = MeshChamfer({"normalize": "none"}).compute(prediction, ground_truth)

    assert canonical == pytest.approx(0.0, abs=1e-6)
    assert raw > 1.0


def test_requires_ground_truth() -> None:
    with pytest.raises(ValueError, match="requires ground truth"):
        MeshChamfer().compute(SceneOutput(layout=None, objects=()), None)  # ty: ignore[missing-argument]
