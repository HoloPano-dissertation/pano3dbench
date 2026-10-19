import pytest

pytest.importorskip("front3d_pano")

np = pytest.importorskip("numpy")
pytest.importorskip("shapely")
from shapely.geometry import Polygon, Point, LineString, mapping


def layout(polygon):
    return {"polygon": mapping(polygon), "floor_z": 0., "ceiling_z": 3.}


def test_rectangle_ranges_and_boundaries(monkeypatch):
    from layout_targets import polygon_targets
    target = polygon_targets(layout(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])), [1, 1, 1.5], 4, 2)
    np.testing.assert_allclose(target["ranges"], np.sqrt(2), atol=1e-6)
    np.testing.assert_allclose(target["boundary"][0], -target["boundary"][1], atol=1e-6)
    assert target["corner"].shape == (1, 4)


def test_concave_polygon_rays_hit_first_wall(monkeypatch):
    from layout_targets import polygon_targets
    polygon = Polygon([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)])
    camera = np.array([0.5, 2.5, 1.5])
    target = polygon_targets(layout(polygon), camera, 64, 32)
    for x, distance in enumerate(target["ranges"]):
        angle = ((x+0.5)/64-0.5)*2*np.pi
        direction = np.array([np.sin(angle), np.cos(angle)])
        ray = LineString([camera[:2], camera[:2]+direction*100])
        expected = ray.intersection(polygon.boundary).distance(Point(camera[:2]))
        assert distance == pytest.approx(expected, abs=1e-6)


def test_holes_occlude_far_wall(monkeypatch):
    from layout_targets import ray_ranges
    polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)],
                      holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]])
    ranges = ray_ranges(polygon, [0.5, 2], [[1, 0], [-1, 0], [0, 1]])
    np.testing.assert_allclose(ranges, [0.5, 0.5, 2.0])


def test_camera_in_missing_corner_rejected(monkeypatch):
    from layout_targets import polygon_targets
    polygon = Polygon([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)])
    with pytest.raises(ValueError, match="actual polygon"):
        polygon_targets(layout(polygon), [2, 2, 1.5])


def test_native_corner_order_preserves_polygon_adjacency(monkeypatch):
    from layout_targets import native_polygon_corners
    polygon = Polygon([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)])
    camera = np.array([0.5, 2.5, 1.5])
    corners = native_polygon_corners(layout(polygon), camera, 1024, 512)
    longitude = ((corners[::2, 0]+0.5)/1024-0.5)*2*np.pi
    lat = ((corners[1::2, 1]+0.5)/512-0.5)*np.pi
    radius = camera[2]/np.tan(lat)
    xy = np.column_stack((np.sin(longitude)*radius, np.cos(longitude)*radius))+camera[:2]
    reconstructed = Polygon(xy)
    assert reconstructed.symmetric_difference(polygon).area < 1e-8

