from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pano3dbench.detection2d.average_precision import AveragePrecision, Labelled
from pano3dbench.detection2d.masks import PanoramaBox, PanoramaMask
from pano3dbench.detection2d.panoramas import (
    PanoramaObject,
    load_coco_ground_truth,
    read_detections,
    read_sample_ids,
)
from pano3dbench.evaluation.result_writer import class_section


def evaluate_detections(
    ground_truth: dict[str, list[PanoramaObject]],
    detections: dict[str, list[PanoramaObject]],
    sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if sample_ids is None:
        sample_ids = sorted(ground_truth)
    missing = [sample_id for sample_id in sample_ids if sample_id not in ground_truth]
    if missing:
        raise ValueError(
            f"{len(missing)} samples have no ground truth, e.g. {missing[0]}"
        )
    unknown = sorted(set(detections) - set(sample_ids))
    boxes: AveragePrecision[PanoramaBox] = AveragePrecision(PanoramaBox.iou)
    masks: AveragePrecision[PanoramaMask] = AveragePrecision(PanoramaMask.iou)
    for sample_id in sample_ids:
        truths = ground_truth[sample_id]
        found = detections.get(sample_id, [])
        boxes.add_image(
            [Labelled(item.label, item.box, item.score) for item in found],
            [Labelled(item.label, item.box, on_seam=item.box.crosses_seam) for item in truths],
        )
        masks.add_image(
            [Labelled(item.label, item.mask, item.score) for item in found],
            [Labelled(item.label, item.mask, on_seam=item.box.crosses_seam) for item in truths],
        )
    return {
        "samples": len(sample_ids),
        "samples_without_detections": sum(
            1 for sample_id in sample_ids if not detections.get(sample_id)
        ),
        "ignored_detection_samples": len(unknown),
        "box": boxes.summary(),
        "mask": masks.summary(),
    }


def run_detection_evaluation(
    coco: str | Path,
    detections: str | Path,
    output_dir: str | Path,
    manifest: str | Path | None = None,
) -> dict[str, Any]:
    ground_truth = load_coco_ground_truth(coco)
    found = dict(read_detections(detections))
    sample_ids = read_sample_ids(manifest) if manifest is not None else None
    summary = evaluate_detections(ground_truth, found, sample_ids)
    summary["inputs"] = {
        "coco": str(coco),
        "detections": str(detections),
        "manifest": str(manifest) if manifest is not None else None,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "report.md").write_text(detection_report(summary), encoding="utf-8")
    return summary


def detection_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 2D detection",
        "",
        f"Samples: {summary['samples']}, without detections: "
        f"{summary['samples_without_detections']}.",
        "",
        "| Region | AP | AP50 | AP75 | Recall50 | GT objects | Predictions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for region in ("box", "mask"):
        values = summary[region]
        lines.append(
            f"| {region} | {values['ap']:.3f} | {values['ap50']:.3f} | "
            f"{values['ap75']:.3f} | {values['recall50']:.3f} | "
            f"{values['gt_objects']} | {values['predictions']} |"
        )
    seam = summary["box"]["seam"]
    lines.extend(
        [
            "",
            f"Objects crossing the panorama seam: {seam['gt_objects']}, box recall at "
            f"IoU 0.5 {seam['recall50']:.3f}; other objects: {seam['rest_gt_objects']}, "
            f"recall {seam['rest_recall50']:.3f}.",
        ]
    )
    for region in ("box", "mask"):
        classes = summary[region]["classes"]
        if len(classes) > 1:
            lines.extend(class_section(f"{region}, AP at IoU 0.50:0.95", classes))
            lines.extend(
                class_section(
                    f"{region}, AP at IoU 0.50",
                    {
                        label: {**values, "ap": values["ap50"]}
                        for label, values in classes.items()
                    },
                )
            )
    return "\n".join(lines) + "\n"
