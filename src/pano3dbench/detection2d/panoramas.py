from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pano3dbench.detection2d.masks import PanoramaBox, PanoramaMask, box_of


@dataclass(frozen=True)
class PanoramaObject:
    label: str
    mask: PanoramaMask
    box: PanoramaBox
    score: float = 1.0


def object_from_mask(label: str, mask: PanoramaMask, score: float = 1.0) -> PanoramaObject | None:
    box = box_of(mask)
    if box is None:
        return None
    return PanoramaObject(label, mask, box, score)


def load_coco_ground_truth(path: str | Path) -> dict[str, list[PanoramaObject]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in document["categories"]}
    images = {int(item["id"]): item for item in document["images"]}
    pieces: dict[str, dict[str, list[tuple[str, PanoramaMask]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sample_ids: list[str] = []
    for image in document["images"]:
        sample_id = str(image["sample_id"])
        if sample_id not in pieces:
            sample_ids.append(sample_id)
            pieces[sample_id]
    for annotation in document["annotations"]:
        image = images[int(annotation["image_id"])]
        mask = PanoramaMask.from_rle(annotation["segmentation"])
        frame = int(image.get("panorama_frame", 0))
        if frame:
            mask = mask.rolled(-(mask.width // 2) * frame)
        instance = str(annotation.get("source_file") or f"annotation-{annotation['id']}")
        pieces[str(image["sample_id"])][instance].append(
            (categories[int(annotation["category_id"])], mask)
        )
    result: dict[str, list[PanoramaObject]] = {}
    for sample_id in sample_ids:
        objects: list[PanoramaObject] = []
        for instance, parts in pieces[sample_id].items():
            labels = {label for label, _ in parts}
            if len(labels) != 1:
                raise ValueError(
                    f"Instance {instance} of {sample_id} has several classes: {sorted(labels)}"
                )
            mask = parts[0][1]
            for _, piece in parts[1:]:
                mask = mask.union(piece)
            item = object_from_mask(parts[0][0], mask)
            if item is not None:
                objects.append(item)
        result[sample_id] = objects
    return result


def read_detections(path: str | Path) -> Iterator[tuple[str, list[PanoramaObject]]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                yield str(record["sample_id"]), list(_detections(record["objects"]))


def _detections(objects: Iterable[Mapping[str, Any]]) -> Iterator[PanoramaObject]:
    for item in objects:
        found = object_from_mask(
            str(item["label"]),
            PanoramaMask.from_rle(item["segmentation"]),
            float(item["score"]),
        )
        if found is not None:
            yield found


def read_sample_ids(manifest: str | Path) -> list[str]:
    with Path(manifest).open(encoding="utf-8") as handle:
        return [str(json.loads(line)["sample_id"]) for line in handle if line.strip()]
