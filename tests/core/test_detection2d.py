import json
from pathlib import Path

import pytest

from pano3dbench.cli import main
from pano3dbench.detection2d import (
    AveragePrecision,
    Labelled,
    PanoramaBox,
    PanoramaMask,
    box_of,
    evaluate_detections,
    load_coco_ground_truth,
    object_from_mask,
)

HEIGHT, WIDTH = 4, 8


def mask_of(cells: set[tuple[int, int]], height: int = HEIGHT, width: int = WIDTH) -> PanoramaMask:
    flags = [0] * (height * width)
    for row, column in cells:
        flags[column * height + row] = 1
    counts: list[int] = []
    current, run = 0, 0
    for flag in flags:
        if flag == current:
            run += 1
        else:
            counts.append(run)
            current, run = flag, 1
    counts.append(run)
    return PanoramaMask.from_rle({"size": [height, width], "counts": counts})


def block(rows: range, columns: range) -> set[tuple[int, int]]:
    return {(row, column % WIDTH) for row in rows for column in columns}


def test_rle_round_trip_and_area() -> None:
    mask = mask_of(block(range(1, 3), range(2, 5)))
    assert mask.area == 6
    assert PanoramaMask.from_rle(mask.to_rle()) == mask


def test_rle_with_wrong_pixel_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="covers 5 pixels"):
        PanoramaMask.from_rle({"size": [2, 3], "counts": [2, 3]})


def test_rolling_moves_columns_around_the_seam() -> None:
    mask = mask_of(block(range(0, 2), range(6, 8)))
    assert mask.rolled(3) == mask_of(block(range(0, 2), range(9, 11)))
    assert mask.rolled(-WIDTH) == mask


def test_mask_iou_does_not_care_about_the_seam() -> None:
    first = mask_of(block(range(0, 4), range(6, 10)))
    second = mask_of(block(range(0, 4), range(7, 11)))
    assert first.intersection_area(second) == 12
    assert first.iou(second) == pytest.approx(12 / 20)


def test_box_of_a_mask_on_the_seam_runs_past_the_right_edge() -> None:
    box = box_of(mask_of(block(range(1, 3), range(7, 10))))
    assert box == PanoramaBox(7, 1, 3, 2, WIDTH)
    assert box.crosses_seam
    assert box_of(mask_of(block(range(0, 1), range(2, 4)))) == PanoramaBox(2, 0, 2, 1, WIDTH)


def test_box_of_a_mask_with_a_gap_takes_the_shorter_arc() -> None:
    box = box_of(mask_of(block(range(0, 1), range(0, 1)) | block(range(3, 4), range(6, 7))))
    assert box == PanoramaBox(6, 0, 3, 4, WIDTH)


def test_box_iou_counts_overlap_across_the_seam() -> None:
    across = PanoramaBox(7, 0, 3, 2, WIDTH)
    same_shifted = PanoramaBox(0, 0, 2, 2, WIDTH)
    assert across.iou(same_shifted) == pytest.approx(4 / 6)
    assert across.iou(PanoramaBox(3, 0, 2, 2, WIDTH)) == 0.0


def _coco(tmp_path: Path) -> Path:
    whole = mask_of(block(range(0, 2), range(1, 3)))
    across_rolled = mask_of(block(range(2, 4), range(3, 6)))
    document = {
        "categories": [{"id": 1, "name": "bed"}, {"id": 2, "name": "lamp"}],
        "images": [
            {"id": 1, "sample_id": "h/r/0", "panorama_frame": 0, "width": WIDTH, "height": HEIGHT},
            {"id": 2, "sample_id": "h/r/0", "panorama_frame": 1, "width": WIDTH, "height": HEIGHT},
            {"id": 3, "sample_id": "h/r/1", "panorama_frame": 0, "width": WIDTH, "height": HEIGHT},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": whole.to_rle(), "source_file": "a.glb"},
            {"id": 2, "image_id": 2, "category_id": 2, "segmentation": across_rolled.to_rle(), "source_file": "b.glb"},
        ],
    }
    path = tmp_path / "test.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_coco_ground_truth_brings_the_turned_frame_back(tmp_path: Path) -> None:
    truth = load_coco_ground_truth(_coco(tmp_path))
    assert sorted(truth) == ["h/r/0", "h/r/1"]
    assert truth["h/r/1"] == []
    labels = {item.label: item for item in truth["h/r/0"]}
    assert labels["bed"].box == PanoramaBox(1, 0, 2, 2, WIDTH)
    assert labels["lamp"].box == PanoramaBox(7, 2, 3, 2, WIDTH)
    assert labels["lamp"].mask == mask_of(block(range(2, 4), range(7, 10)))


def test_perfect_detections_give_ap_one(tmp_path: Path) -> None:
    truth = load_coco_ground_truth(_coco(tmp_path))
    summary = evaluate_detections(truth, truth)
    for region in ("box", "mask"):
        assert summary[region]["ap"] == pytest.approx(1.0)
        assert summary[region]["classes"]["lamp"]["gt_objects"] == 1
    assert summary["box"]["seam"] == {
        "gt_objects": 1,
        "recall50": 1.0,
        "rest_gt_objects": 1,
        "rest_recall50": 1.0,
    }


def test_two_halves_of_a_seam_object_are_one_hit_and_one_miss(tmp_path: Path) -> None:
    truth = load_coco_ground_truth(_coco(tmp_path))
    halves = [
        object_from_mask("lamp", mask_of(block(range(2, 4), range(7, 8))), 0.9),
        object_from_mask("lamp", mask_of(block(range(2, 4), range(0, 2))), 0.8),
    ]
    found = {"h/r/0": [item for item in halves if item is not None]}
    lamp = evaluate_detections(truth, found)["box"]["classes"]["lamp"]
    assert lamp["predictions"] == 2
    assert lamp["recall50"] == 1.0
    assert lamp["ap50"] == pytest.approx(0.5)
    assert lamp["ap"] == pytest.approx(0.2)


def test_average_precision_follows_coco_interpolation() -> None:
    def iou(first: int, second: int) -> float:
        return 1.0 if first == second else 0.0

    scorer: AveragePrecision[int] = AveragePrecision(iou)
    scorer.add_image(
        [Labelled("chair", 1, 0.9), Labelled("chair", 9, 0.8), Labelled("chair", 2, 0.7)],
        [Labelled("chair", 1), Labelled("chair", 2)],
    )
    scorer.add_image([], [Labelled("chair", 3), Labelled("sofa", 4)])
    summary = scorer.summary()
    chair = summary["classes"]["chair"]
    expected = (34 * 1.0 + 33 * (2 / 3)) / 101
    assert chair["ap"] == pytest.approx(expected)
    assert chair["recall50"] == pytest.approx(2 / 3)
    assert summary["classes"]["sofa"]["ap"] == 0.0
    assert summary["ap"] == pytest.approx(expected / 2)


def test_detect2d_command_writes_summary_and_report(tmp_path: Path) -> None:
    coco = _coco(tmp_path)
    truth = load_coco_ground_truth(coco)
    detections = tmp_path / "detections.jsonl"
    detections.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "objects": [
                        {"label": item.label, "score": 0.9, "segmentation": item.mask.to_rle()}
                        for item in items
                    ],
                }
            )
            + "\n"
            for sample_id, items in truth.items()
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"sample_id": "h/r/0"}\n', encoding="utf-8")
    output = tmp_path / "out"

    assert main(["detect2d", "--coco", str(coco), "--detections", str(detections),
                 "--manifest", str(manifest), "--output", str(output)]) == 0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["samples"] == 1
    assert summary["box"]["ap"] == pytest.approx(1.0)
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "| box | 1.000 | 1.000 | 1.000 | 1.000 | 2 | 2 |" in report
    assert "## Per-class AP: mask, AP at IoU 0.50" in report
    assert "Objects crossing the panorama seam: 1" in report
