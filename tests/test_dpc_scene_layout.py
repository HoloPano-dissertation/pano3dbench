import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

TRAINING = Path(__file__).resolve().parents[1] / "tools" / "3dfront_training"


@pytest.fixture(scope="module")
def naming():
    spec = importlib.util.spec_from_file_location(
        "export_dpc_scenes_naming", TRAINING / "export_dpc_scenes.py")
    import sys, types
    for name in ("utils", "utils.igibson_utils", "utils.relation_utils",
                 "utils.render_utils", "h5py"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["utils.igibson_utils"].IGScene = type("IGScene", (), {})
    sys.modules["utils.relation_utils"].RelationOptimization = type("R", (), {})
    sys.modules["utils.render_utils"].seg2obj = lambda *a, **k: None
    sys.modules["utils.render_utils"].is_obj_valid = lambda *a, **k: True
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        pytest.skip("the exporter is not importable here: %s" % error)
    return module


@pytest.fixture(scope="module")
def scenes():
    spec = importlib.util.spec_from_file_location(
        "scene_layout", TRAINING / "scene_layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def square_layout(points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))):
    ring = [list(p) for p in points] + [list(points[0])]
    return {"polygon": {"type": "Polygon", "coordinates": [ring]},
            "floor_z": 0.0, "ceiling_z": 2.6}


def dpc_walls(layout):
    """Front faces exactly as DPC's wall_bdb3d_from_manhattan_world_layout indexes them."""
    n = len(layout) // 2
    return [layout[(i, (i + 1) % n, i + n, (i + 1) % n + n), :] for i in range(n)]


def signed_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def test_floor_ring_comes_first_then_the_ceiling_ring(scenes):
    corners = scenes.manhattan_world(square_layout())
    assert corners.shape == (8, 3)
    assert (corners[:4, 2] == 0.0).all()
    assert (corners[4:, 2] == 2.6).all()
    assert np.allclose(corners[:4, :2], corners[4:, :2])


def test_the_plan_runs_clockwise_like_igibson(scenes):
    corners = scenes.manhattan_world(square_layout())
    assert signed_area(corners[:4, :2]) == pytest.approx(-12.0)


def test_every_dpc_wall_is_one_contour_edge_from_floor_to_ceiling(scenes):
    layout = square_layout(((0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)))
    corners = scenes.manhattan_world(layout)
    plan = corners[:6, :2]
    edges = {tuple(sorted((tuple(plan[i]), tuple(plan[(i + 1) % 6])))) for i in range(6)}
    for face in dpc_walls(corners):
        floor_a, floor_b, ceiling_a, ceiling_b = face
        assert floor_a[2] == floor_b[2] == 0.0
        assert ceiling_a[2] == ceiling_b[2] == 2.6
        assert np.allclose(floor_a[:2], ceiling_a[:2]) and np.allclose(floor_b[:2], ceiling_b[:2])
        assert tuple(sorted((tuple(floor_a[:2]), tuple(floor_b[:2])))) in edges


def test_the_dpc_plan_polygon_is_the_room(scenes):
    layout = square_layout(((0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)))
    corners = scenes.manhattan_world(layout)
    assert abs(signed_area(corners[:6, :2])) == pytest.approx(5.0)


def horizonnet_floor_plan(pixels, camera_height):
    floor = np.asarray(pixels)[1::2]
    u = ((floor[:, 0] + 0.5) / 1024 - 0.5) * 2 * np.pi
    v = -((floor[:, 1] + 0.5) / 512 - 0.5) * np.pi
    c = -camera_height / np.tan(v)
    return np.column_stack((c * np.sin(u) + 0.5 - 0.5, -c * np.cos(u) + 0.5 - 0.5))


CAMERA = np.array([1.0, 1.0, 1.4])


def test_pixel_pairs_share_a_column_with_the_ceiling_above(scenes):
    corners = scenes.manhattan_world(square_layout())
    pixels = scenes.manhattan_pix(corners, CAMERA, 1024, 512)
    assert (pixels[0::2, 0] == pixels[1::2, 0]).all()
    assert (pixels[0::2, 1] < pixels[1::2, 1]).all()


def test_pixel_corners_start_leftmost_and_rise_in_a_convex_room(scenes):
    corners = scenes.manhattan_world(square_layout())
    pixels = scenes.manhattan_pix(corners, CAMERA, 1024, 512)
    assert pixels[0, 0] == pixels[0::2, 0].min()
    assert (np.diff(pixels[0::2, 0]) > 0).all()


def test_a_hidden_corner_keeps_its_wall_order_instead_of_being_sorted(scenes):
    layout = square_layout(((0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)))
    corners = scenes.manhattan_world(layout)
    pixels = scenes.manhattan_pix(corners, np.array([2.5, 0.5, 1.4]), 1024, 512)
    columns = pixels[0::2, 0]
    assert (np.diff(columns) < 0).any()
    assert len(columns) == 6


def test_horizonnet_evaluation_recovers_exactly_our_plan(scenes):
    layout = square_layout(((0, 0), (9, 0), (9, 1), (1, 1), (1, 3), (0, 3)))
    corners = scenes.manhattan_world(layout)
    camera = np.array([0.5, 2.5, 1.6])
    plan = horizonnet_floor_plan(scenes.manhattan_pix(corners, camera, 1024, 512), camera[2])
    from shapely.geometry import Polygon
    recovered = Polygon(plan)
    assert recovered.is_valid
    assert recovered.area == pytest.approx(11.0, rel=1e-9)


def dpc_obj2frame(points, bdb3d):
    return (bdb3d["basis"] @ (points * bdb3d["size"]).T).T + bdb3d["centroid"]


def test_a_rotated_box_lands_where_the_ground_truth_puts_it(scenes):
    theta = np.radians(30.0)
    axis_x = np.array([np.cos(theta), np.sin(theta)])
    axis_y = np.array([-axis_x[1], axis_x[0]])
    box = {"center": [1.0, 2.0, 0.5], "size": [2.0, 1.0, 0.8],
           "basis": [[axis_x[0], axis_x[1], 0.0], [axis_y[0], axis_y[1], 0.0], [0.0, 0.0, 1.0]]}
    camera = np.array([0.3, -0.2, 1.4])
    signs = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    ours = np.asarray(box["center"]) + camera + (signs * box["size"]) @ np.asarray(box["basis"])
    dpc = dpc_obj2frame(signs, scenes.dpc_bdb3d(box, camera))
    assert np.linalg.norm(ours[:, None] - dpc[None], axis=2).min(axis=1).max() < 1e-9


def test_the_dpc_box_axes_are_right_handed(scenes):
    box = {"center": [0, 0, 0], "size": [1, 1, 1],
           "basis": [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]}
    assert np.linalg.det(scenes.dpc_bdb3d(box, [0, 0, 0])["basis"]) == pytest.approx(1.0)


def test_rounding_noise_never_splits_a_pair(scenes):
    """The two corners of a pair are one plan position, so one column."""
    noisy = np.array([[126.056000, 112.3], [126.055992, 431.1],
                      [ 83.200000, 120.0], [ 83.200008, 400.0]])
    snapped = scenes.share_one_column(noisy)
    assert (snapped[0::2, 0] == snapped[1::2, 0]).all()
    assert list(snapped[1::2, 0]) == [126.056, 83.2]
    assert list(snapped[:, 1]) == list(noisy[:, 1])


def test_a_camera_is_named_by_room_and_view_not_view_alone(naming):
    """The dump writes <house>/<name>, so two rooms of one house must not collide."""
    first = naming.camera_name("Bedroom-1", "0")
    second = naming.camera_name("Kitchen-2", "0")
    assert first != second
    assert "/" not in first and "/" not in second
    assert first == "Bedroom-1__0"
