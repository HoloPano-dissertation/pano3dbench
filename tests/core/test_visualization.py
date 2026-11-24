import json
import math
from pathlib import Path

import pytest

from pano3dbench.cli import main
from pano3dbench.geometry.boxes import BoundingBox3D, OrientedBox3D
from pano3dbench.visualization import (
    OverlayOptions,
    Panorama,
    box_polylines,
    overlay_shapes,
    room_polylines,
)
from pano3dbench.visualization.projection import BOX_EDGES, box_corners

PANORAMA = Panorama(1024, 512)


def test_projection_follows_the_dataset_convention() -> None:
    assert PANORAMA.project((0.0, 1.0, 0.0)) == pytest.approx((511.5, 255.5))
    assert PANORAMA.project((1.0, 0.0, 0.0)) == pytest.approx((767.5, 255.5))
    assert PANORAMA.project((-1.0, 0.0, 0.0)) == pytest.approx((255.5, 255.5))
    assert PANORAMA.project((0.0, 1.0, 1.0)) == pytest.approx((511.5, 127.5))
    assert PANORAMA.project((0.0, 1.0, -1.0)) == pytest.approx((511.5, 383.5))


def test_box_edges_join_neighbouring_corners() -> None:
    box = OrientedBox3D((0.0, 3.0, 0.0), (1.0, 2.0, 4.0), ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    corners = box_corners(box)
    lengths = sorted(round(math.dist(corners[a], corners[b]), 9) for a, b in BOX_EDGES)
    assert lengths == [1.0] * 4 + [2.0] * 4 + [4.0] * 4


def test_an_edge_across_the_seam_is_split_into_two_continuous_pieces() -> None:
    pieces = PANORAMA.polylines((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0))
    assert len(pieces) == 2
    for piece in pieces:
        steps = [abs(b[0] - a[0]) for a, b in zip(piece, piece[1:])]
        assert max(steps) < 50
    assert pieces[0][-1][0] < 0
    assert pieces[1][0][0] > PANORAMA.width - 1


def test_an_edge_in_front_stays_one_piece() -> None:
    pieces = PANORAMA.polylines((-1.0, 2.0, 0.0), (1.0, 2.0, 0.0))
    assert len(pieces) == 1
    assert pieces[0][0][0] < 511.5 < pieces[0][-1][0]


def test_a_box_gives_one_polyline_per_edge_when_off_the_seam() -> None:
    lines = box_polylines(PANORAMA, BoundingBox3D((-0.5, 2.0, -1.0), (0.5, 3.0, 0.0)))
    assert len(lines) == 12


def test_room_outline_draws_floor_ceiling_and_corners() -> None:
    square = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0), (-2.0, -2.0)]
    lines = room_polylines(PANORAMA, square, -1.5, 1.0)
    covered = {round(point[0]) for line in lines for point in line}
    assert len(lines) >= 12
    assert min(covered) <= 0 and max(covered) >= PANORAMA.width - 1


def _scene(score: float = 1.0, with_room: bool = False) -> dict:
    document = {
        "layout": {"min_corner": [-2.0, -2.0, -1.5], "max_corner": [2.0, 2.0, 1.0]},
        "objects": [
            {
                "object_id": "a",
                "label": "bed",
                "score": score,
                "bbox": {"center": [0.0, 1.0, -1.0], "size": [1.0, 2.0, 0.5],
                         "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            }
        ],
    }
    if with_room:
        document["metadata"] = {
            "camera_location_world": [10.0, 10.0, 1.5],
            "layout_geometry_world": {
                "polygon": {
                    "type": "Polygon",
                    "coordinates": [[[8, 8], [12, 8], [12, 11], [9, 11], [9, 12], [8, 12], [8, 8]]],
                }
            },
        }
    return document


def test_overlay_uses_the_room_polygon_when_the_ground_truth_has_one() -> None:
    plain, _ = overlay_shapes(PANORAMA, _scene(), None)
    room, _ = overlay_shapes(PANORAMA, _scene(with_room=True), None)
    assert len(room) > len(plain)


def test_overlay_drops_predictions_below_the_score_and_labels_the_rest() -> None:
    strokes, labels = overlay_shapes(
        PANORAMA, None, _scene(score=0.3), OverlayOptions(min_score=0.5, layout=False)
    )
    assert strokes == [] and labels == []
    strokes, labels = overlay_shapes(
        PANORAMA, _scene(), _scene(score=0.8), OverlayOptions(layout=False)
    )
    assert [label.text for label in labels] == ["bed", "bed 0.80"]
    assert {stroke.color for stroke in strokes} == {(40, 200, 80), (240, 60, 50)}


def test_draw_command_renders_a_manifest(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    panorama = tmp_path / "rgb" / "h" / "r" / "0.png"
    panorama.parent.mkdir(parents=True)
    image_module.new("RGB", (256, 128)).save(panorama)
    truth = tmp_path / "gt.json"
    truth.write_text(json.dumps(_scene(with_room=True)), encoding="utf-8")
    predictions = tmp_path / "pred" / "h" / "r"
    predictions.mkdir(parents=True)
    (predictions / "0.json").write_text(json.dumps(_scene(score=0.9)), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"sample_id": "h/r/0", "input": str(panorama), "ground_truth": str(truth)}) + "\n",
        encoding="utf-8",
    )

    assert main(["draw", "--manifest", str(manifest), "--predictions", str(tmp_path / "pred"),
                 "--output", str(tmp_path / "out")]) == 0

    with image_module.open(tmp_path / "out" / "h" / "r" / "0.png") as drawn:
        colors = {color for _, color in drawn.convert("RGB").getcolors(maxcolors=1 << 16)}
    assert (40, 200, 80) in colors
    assert (240, 60, 50) in colors
