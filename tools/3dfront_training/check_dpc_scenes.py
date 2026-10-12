#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, shape

from external.HorizonNet.misc import post_proc
from utils.layout_utils import (manhattan_2d_from_manhattan_world_layout,
                                wall_bdb3d_from_manhattan_world_layout)
from utils.transform_utils import bdb3d_corners

TOLERANCE = 1e-3  # m
MESHES_SEEN = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("scenes_root", type=Path)
    parser.add_argument("--sample", type=int, default=300, help="scenes to check; 0 for all")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def ground_truth_corners(box, camera):
    signs = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    return (np.asarray(box["center"]) + np.asarray(camera)
            + (signs * np.asarray(box["size"])) @ np.asarray(box["basis"]))


def set_distance(first, second):
    return float(np.linalg.norm(first[:, None] - second[None], axis=2).min(axis=1).max())


def check_scene(data, truth):
    problems = []
    metadata = truth["metadata"]
    layout = metadata["layout_geometry_world"]
    world = np.asarray(data["layout"]["manhattan_world"], dtype=float)
    count = len(world) // 2
    plan = shape(layout["polygon"])
    ring = np.asarray(plan.exterior.coords)[:-1]
    if count != len(ring):
        problems.append(f"{count} DPC corners for a {len(ring)}-corner contour")
    height = float(layout["ceiling_z"]) - float(layout["floor_z"])
    walls = wall_bdb3d_from_manhattan_world_layout(world)
    edges = sorted(float(np.hypot(*(ring[(i + 1) % len(ring)] - ring[i]))) for i in range(len(ring)))
    widths, heights = [], []
    for wall in walls:
        size = np.asarray(wall["bdb3d"]["size"], dtype=float)
        widths.append(float(size[0]))
        heights.append(float(size[2]))
    if heights and max(abs(h - height) for h in heights) > TOLERANCE:
        problems.append(f"wall heights {min(heights):.3f}..{max(heights):.3f} m, room {height:.3f} m")
    if len(widths) == len(edges) and max(abs(a - b) for a, b in zip(sorted(widths), edges)) > TOLERANCE:
        problems.append("wall widths differ from contour edges")
    outer = Polygon(plan.exterior)
    dpc_plan = Polygon(manhattan_2d_from_manhattan_world_layout(world))
    if not dpc_plan.is_valid or abs(dpc_plan.area - outer.area) > TOLERANCE * max(outer.area, 1.0):
        problems.append(f"DPC plan area {dpc_plan.area:.3f} vs outer contour {outer.area:.3f}")
    if dpc_plan.exterior.is_ccw:
        problems.append("DPC plan runs counter-clockwise")

    pixels = np.asarray(data["layout"]["manhattan_pix"], dtype=float)
    if len(pixels) != len(world):
        problems.append("manhattan_pix and manhattan_world differ in length")
    elif not ((pixels[0::2, 0] == pixels[1::2, 0]).all() and (pixels[0::2, 1] < pixels[1::2, 1]).all()):
        problems.append("manhattan_pix pairs are not (ceiling, floor) in one column")
    else:
        # HorizonNet's evaluation rebuilds the plan from the floor pixels; it must get ours back.
        camera_height = float(data["camera"]["pos"][2]) - float(world[0, 2])
        rebuilt = Polygon(post_proc.np_coor2xy(pixels[1::2], -camera_height, 1024, 512,
                                               floorW=1, floorH=1))
        if not rebuilt.is_valid or abs(rebuilt.area - outer.area) > TOLERANCE * max(outer.area, 1.0):
            problems.append(f"HorizonNet rebuilds a plan of {rebuilt.area:.3f} "
                            f"(valid {rebuilt.is_valid}) from manhattan_pix, contour {outer.area:.3f}")

    camera = metadata["camera_location_world"]
    truth_corners = [ground_truth_corners(o["bbox"], camera) for o in truth["objects"]]
    worst = 0.0
    for obj in data["objs"]:
        corners = np.asarray(bdb3d_corners(obj["bdb3d"]), dtype=float)
        worst = max(worst, min(set_distance(corners, t) for t in truth_corners))
    if worst > TOLERANCE:
        problems.append(f"object box off its ground truth by {worst:.3f} m")
    for obj in data["objs"]:
        mesh = Path(obj["model_path"]) / "mesh_watertight.ply"  # as DPC opens it
        if mesh not in MESHES_SEEN:
            if not mesh.is_file():
                problems.append(f"no ground-truth mesh {mesh}")
                break
            MESHES_SEEN.add(mesh)
    relation = data.get("relation") or {}
    for key in ("obj_wall_rot", "obj_wall_tch"):
        if key in relation and np.asarray(relation[key]).shape[-1] != count:
            problems.append(f"{key} has {np.asarray(relation[key]).shape[-1]} walls, layout {count}")
    return problems, worst, len(plan.interiors)


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    entries = []
    for split in ("train", "val", "test"):
        for line in (root / "manifests_gt" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                house, room, view = record["sample_id"].split("/")
                pkl = args.scenes_root / split / house / room / f"{room}__{view}" / "data.pkl"
                entries.append((pkl, Path(record["ground_truth"])))
    if args.sample:
        entries = random.Random(args.seed).sample(entries, min(args.sample, len(entries)))
    failures, missing, checked, worst, with_holes = [], 0, 0, 0.0, []
    for pkl, truth_path in entries:
        if not pkl.is_file():
            missing += 1  # scenes without usable objects are not exported
            continue
        with pkl.open("rb") as handle:
            data = pickle.load(handle)
        problems, error, holes = check_scene(data, json.loads(truth_path.read_text(encoding="utf-8")))
        checked += 1
        worst = max(worst, error)
        if holes:
            with_holes.append(str(pkl))
        if problems:
            failures.append({"scene": str(pkl), "problems": problems})
    status = {"ready": not failures and checked > 0, "checked": checked,
              "not_exported": missing, "failed": len(failures),
              "worst_object_error_m": worst,
              "scenes_with_plan_holes_dropped_by_dpc": len(with_holes),
              "holes_examples": with_holes[:10], "failures": failures[:20]}
    print(json.dumps(status, indent=2, ensure_ascii=False))
    sys.exit(0 if status["ready"] else 1)


if __name__ == "__main__":
    main()
