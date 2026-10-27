#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner_protocol as protocol  # noqa: E402
import scene_payload as payload  # noqa: E402


def camera_position(data) -> list:
    camera = data.get("camera")
    position = dict.get(camera, "pos") if isinstance(camera, dict) else None
    if position is None:
        return [0.0, 0.0, 0.0]
    return [float(value) for value in list(position)[:3]]


def axes_as_rows(basis) -> list:
    rows = [[float(value) for value in row] for row in basis]
    return [[rows[row][column] for row in range(3)] for column in range(3)]


def convert_data_pkl(data) -> dict:
    origin = camera_position(data)
    objects = []
    for item in data.get("objs", []):
        bdb3d = item.get("bdb3d") or {}
        size = bdb3d.get("size")
        centroid = bdb3d.get("centroid")
        basis = bdb3d.get("basis")
        if size is None or centroid is None or basis is None:
            continue
        entry = {
            "label": item.get("classname", "object"),
            "score": float(item.get("score", 1.0)),
            "centroid": [float(value) - offset for value, offset in zip(centroid, origin)],
            "basis": axes_as_rows(basis),
            "coeffs": [float(value) / 2.0 for value in size],
        }
        shape = item.get("mesh_path") or item.get("mesh")
        if isinstance(shape, str):
            entry["shape"] = shape
        objects.append(entry)
    layout = data.get("layout")
    layout_corners = None
    if isinstance(layout, dict) and layout.get("manhattan_world") is not None:
        layout_corners = [[float(value) - offset for value, offset in zip(point, origin)]
                          for point in layout["manhattan_world"]]
    return payload.scene_from_oriented_objects(objects, layout_corners=layout_corners)


def dpc_scene_name(sample_id: str) -> str:
    parts = sample_id.split("/")
    if len(parts) == 3:
        return "%s-%s__%s" % tuple(parts)
    return sample_id


def load_data_pkl(path: str):
    repo_root = os.getcwd()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return data.data if hasattr(data, "data") else data


def prediction_sample_id(path: str) -> str:
    sample_dir = os.path.dirname(path)
    parent = os.path.basename(sample_dir)
    grandparent = os.path.basename(os.path.dirname(sample_dir))
    if grandparent and grandparent != "visualization":
        return f"{grandparent}-{parent}"
    return parent


_INDEX_CACHE: dict = {}


def index_predictions(search_root: str = "out/pano3d") -> list:
    if search_root not in _INDEX_CACHE:
        _INDEX_CACHE[search_root] = sorted(
            glob.glob(
                os.path.join(search_root, "*", "visualization", "**", "data.pkl"),
                recursive=True,
            )
        )
    return _INDEX_CACHE[search_root]


def index_by_sample_id(search_root: str = "out/pano3d") -> dict:
    found: dict = {}
    for path in index_predictions(search_root):
        found.setdefault(prediction_sample_id(path), path)
    return found


def find_data_pkl(
    sample_id: "str | None" = None, search_root: str = "out/pano3d"
) -> "str | None":
    matches = index_predictions(search_root)
    if sample_id:
        for wanted in dict.fromkeys((sample_id, dpc_scene_name(sample_id))):
            for path in matches:
                if prediction_sample_id(path) == wanted:
                    return path
        for path in matches:
            if sample_id in path:
                return path
        return None
    return matches[-1] if matches else None


def run_dpc_inference(
    request: dict,
    config: str,
    pred_root: str = "out/pano3d",
    relation_adjust: str = "True",
) -> str:
    import subprocess

    sample_id = request.get("sample_id")
    existing = find_data_pkl(sample_id, pred_root)
    if existing is not None:
        return existing

    command = [sys.executable, "main.py", config, "--mode", "test"]
    if relation_adjust:
        command += ["--model.scene_gcn.relation_adjust", relation_adjust]
    command += ["--log.path", pred_root]
    subprocess.run(command, check=True)
    _INDEX_CACHE.pop(pred_root, None)
    produced = find_data_pkl(sample_id, pred_root)
    if produced is None:
        raise RuntimeError(
            f"DPC ran, but no data.pkl was found for scene '{sample_id}' — "
            f"check that sample_id matches the scene names in {pred_root}/."
        )
    return produced


def _convert_one(request: dict, args, data_pkl: "str | None" = None) -> dict:
    sample_id = request.get("sample_id")
    path = data_pkl or find_data_pkl(sample_id, args.pred_root)
    if path is None:
        path = run_dpc_inference(
            request, args.config, args.pred_root, args.relation_adjust
        )
    scene = convert_data_pkl(load_data_pkl(path))
    return {
        "layout": scene["layout"],
        "objects": scene["objects"],
        "relations": scene["relations"],
        "metadata": {
            "model": args.model,
            "sample_id": sample_id,
            "data_pkl": str(path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepPanoContext runner")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/pano3d_igibson.yaml")
    parser.add_argument("--data-pkl", help="ready data.pkl (skip inference)")
    parser.add_argument(
        "--pred-root",
        default="out/pano3d",
        help="root of the predictions to score (<root>/<run>/visualization/.../data.pkl), "
        "relative to the DPC repo. THE method selector when scoring existing "
        "inference: DPC writes out/pano3d, and each other configuration of the same "
        "code base writes its own root (out/pano3d_im3d, out/pano3d_total3d, ...), so "
        "pointing this elsewhere is what makes the runner score a different method.",
    )
    parser.add_argument(
        "--model",
        default="dpc",
        help="method label recorded in the prediction metadata (dpc, im3d-pano, ...)",
    )
    parser.add_argument(
        "--relation-adjust",
        default="True",
        help="Scene-GCN relation optimization for a fresh inference run: 'True' for "
        "DPC, 'False' for the Im3D-Pano ablation, empty to let the config decide. "
        "Ignored when predictions already exist under --pred-root.",
    )
    args, _unknown = parser.parse_known_args()

    requests = protocol.read_requests(args.request)
    data_pkl = args.data_pkl if len(requests) == 1 else None
    if not data_pkl:
        protocol.log(f"[dpc_runner] {len(requests)} sample(s), model={args.model}, root={args.pred_root}")
    scenes = [_convert_one(request, args, data_pkl) for request in requests]
    protocol.write_scenes(args.output, scenes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
