#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the 2D detections of DPC (data.pkl) as panorama masks"
    )
    parser.add_argument("manifest", help="manifest.jsonl of the panoramas to export")
    parser.add_argument("output", help="JSON lines file, one panorama per line")
    parser.add_argument(
        "--repo",
        default=os.getcwd(),
        help="cloned DPC repo root (to unpickle utils.*; defaults to the current directory)",
    )
    parser.add_argument(
        "--pred-root",
        default=os.path.join("out", "pano3d"),
        help="predictions root (<root>/<run>/visualization/.../data.pkl); "
        "a relative path is resolved from --repo",
    )
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(here, os.pardir, os.pardir, "runners"))
    import dpc_runner

    pred_root = args.pred_root
    if not os.path.isabs(pred_root):
        pred_root = os.path.join(repo, pred_root)
    manifest = os.path.abspath(args.manifest)
    output = os.path.abspath(args.output)
    os.chdir(repo)

    by_id = {}
    for path in sorted(
        glob.glob(os.path.join(pred_root, "*", "visualization", "**", "data.pkl"), recursive=True)
    ):
        by_id.setdefault(dpc_runner.prediction_sample_id(path), path)

    exported = missing = objects = 0
    with open(output, "w", encoding="utf-8") as out:
        for line in open(manifest, encoding="utf-8"):
            if not line.strip():
                continue
            sample_id = json.loads(line)["sample_id"]
            pkl = by_id.get(sample_id) or by_id.get(dpc_runner.dpc_scene_name(sample_id))
            if pkl is None:
                missing += 1
                continue
            with open(pkl, "rb") as handle:
                data = pickle.load(handle)
            data = data.data if hasattr(data, "data") else data
            record = panorama_record(sample_id, data)
            objects += len(record["objects"])
            out.write(json.dumps(record) + "\n")
            exported += 1

    print(f"exported: {exported} | missing: {missing} | objects: {objects}")
    print(f"detections -> {output}")
    return 0


def panorama_record(sample_id, data):
    camera = data["camera"]
    width, height = int(camera["width"]), int(camera["height"])
    return {
        "sample_id": sample_id,
        "width": width,
        "height": height,
        "objects": [
            {
                "label": str(obj["classname"]),
                "score": float(obj["score"]),
                "segmentation": folded_rle(obj["contour"], height, width),
            }
            for obj in data.get("objs", [])
        ],
    }


def folded_rle(contour, height, width):
    import cv2
    import numpy as np

    points = np.stack(
        [np.asarray(contour["x"], dtype=np.int32), np.asarray(contour["y"], dtype=np.int32)],
        axis=1,
    )
    canvas = np.zeros((height, 2 * width), dtype=np.uint8)
    cv2.fillPoly(canvas, [points.reshape(-1, 1, 2)], 1)
    mask = (canvas[:, :width] | canvas[:, width:]).astype(bool)
    pixels = mask.ravel(order="F").astype(np.uint8)
    changes = np.flatnonzero(np.diff(np.concatenate(([0], pixels, [0]))))
    counts = []
    position = 0
    for start, end in zip(changes[0::2], changes[1::2]):
        counts.extend((int(start - position), int(end - start)))
        position = int(end)
    counts.append(int(pixels.size - position))
    return {"size": [height, width], "counts": counts}


if __name__ == "__main__":
    raise SystemExit(main())
