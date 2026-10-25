#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

W, H = 1024, 512

_cols = np.arange(W) + 0.5
_rows = np.arange(H) + 0.5
_LON = (_cols / W - 0.5) * 2 * np.pi
_LAT = (0.5 - _rows / H) * np.pi
DIR = np.stack([np.cos(_LAT)[:, None] * np.sin(_LON)[None, :],
                np.cos(_LAT)[:, None] * np.cos(_LON)[None, :],
                np.repeat(np.sin(_LAT)[:, None], W, 1)], -1)

BOX_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def project(p):
    p = np.atleast_2d(p)
    r = np.hypot(p[:, 0], p[:, 1])
    x = ((0.5 + np.arctan2(p[:, 0], p[:, 1]) / (2 * np.pi)) % 1) * W - 0.5
    y = (0.5 - np.arctan2(p[:, 2], r) / np.pi) * H - 0.5
    return x, y


def polyline(draw, x, y, fill, width=2):
    pts = list(zip(x.tolist(), y.tolist()))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if abs(x1 - x0) < W / 2:
            draw.line([(x0, y0), (x1, y1)], fill=fill, width=width)


def collinear_vertices(polygon):
    pts = polygon["coordinates"][0][:-1]
    bad = []
    for i in range(len(pts)):
        a, b, c = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[(i + 1) % len(pts)])
        u, v = b - a, c - b
        cross = abs(u[0] * v[1] - u[1] * v[0]) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        if cross < 1e-3 and u @ v > 0:
            bad.append(i)
    return bad


def box_corners(box):
    c, s, B = np.array(box["center"]), np.array(box["size"]), np.array(box["basis"])
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    return c + (signs * s / 2) @ B


def load_splits(exp):
    split = {}
    for name in ("train", "val", "test"):
        for line in (exp / "manifests_gt" / f"{name}.jsonl").read_text().splitlines():
            split[json.loads(line)["metadata"]["room_id"]] = name
    return split


def check_view(exp, split, room, view, render, layout, fake):
    hdir = exp / "outputs" / split / room
    cam = np.array(render["camera_locations"][view])
    with h5py.File(hdir / f"{view}.hdf5", "r") as f:
        depth = f["depth"][()].astype(np.float64)
        seg = f["instance_segmaps"][()]
        attrs = json.loads(f["instance_attribute_maps"][()])
    img = Image.open(exp / "rgb" / split / room / f"{view}.png").convert("RGB")
    scene = json.loads((exp / "ground_truth" / split / room / f"{view}.json").read_text())
    scale = scene["metadata"]["metric_scale"]
    stem = room.replace("/", "__") + f"__{view}"
    dense = np.load(exp / "horizonnet" / split / "label_dense" / f"{stem}.npz")
    txt = exp / "horizonnet" / split / "label_cor" / f"{stem}.txt"
    corners = np.loadtxt(txt).reshape(-1, 2) if txt.is_file() else None

    valid = np.isfinite(depth) & (depth > 0) & (depth < 1e3)
    P = DIR * depth[..., None]
    cz, fz = layout["ceiling_z"] - cam[2], layout["floor_z"] - cam[2]
    tol = 0.01 * (cz - fz)
    on_ceil = valid & (np.abs(P[..., 2] - cz) < tol)
    on_floor = valid & (np.abs(P[..., 2] - fz) < tol)

    pred_c = (dense["boundary"][0] / np.pi + 0.5) * H - 0.5
    pred_f = (dense["boundary"][1] / np.pi + 0.5) * H - 0.5
    true_c = np.where(on_ceil.any(0), H - 1 - np.argmax(on_ceil[::-1], 0), -1).astype(float)
    true_f = np.where(on_floor.any(0), np.argmax(on_floor, 0), -1).astype(float)
    err_c = np.where(true_c >= 0, true_c - pred_c, np.nan)
    err_f = np.where(true_f >= 0, true_f - pred_f, np.nan)
    ceil_frac_top = float(on_ceil[:20][valid[:20]].mean()) if valid[:20].any() else float("nan")

    draw = ImageDraw.Draw(img)
    by_idx = {int(a["idx"]): a for a in attrs}
    objs = {Path(o["attributes"]["source_glb"]).name: o for o in scene["objects"]}
    obj_stats = []
    for inst in np.unique(seg):
        item = by_idx.get(int(inst))
        if not item:
            continue
        o = objs.get(Path(str(item.get("source_file") or "")).name)
        if o is None:
            continue
        m = (seg == inst) & valid
        if m.sum() < 30:
            continue
        pts = P[m] * scale
        B = np.array(o["bbox"]["basis"])
        local = (pts - np.array(o["bbox"]["center"])) @ B.T
        half = np.array(o["bbox"]["size"]) / 2 + 0.03
        inside = float((np.abs(local) <= half).all(1).mean())
        obj_stats.append({"object": o["object_id"], "label": o["label"],
                          "pixels": int(m.sum()), "inside_box": round(inside, 4)})
    for o in scene["objects"]:
        corners3 = box_corners(o["bbox"]) / scale
        st = next((s for s in obj_stats if s["object"] == o["object_id"]), None)
        colour = (255, 0, 255) if st and st["inside_box"] < 0.95 else (0, 220, 255)
        for a, b in BOX_EDGES:
            seg_pts = corners3[a] + np.linspace(0, 1, 48)[:, None] * (corners3[b] - corners3[a])
            x, y = project(seg_pts)
            polyline(draw, x, y, colour, 1)

    ring = render["layout"]["polygon"]["coordinates"][0]
    for a, b in zip(ring, ring[1:]):
        seg_xy = np.array(a) + np.linspace(0, 1, 60)[:, None] * (np.array(b) - np.array(a))
        p3 = np.column_stack((seg_xy - cam[:2], np.full(len(seg_xy), layout["floor_z"] - cam[2])))
        x, y = project(p3)
        for i in range(0, len(x) - 1, 2):
            if abs(x[i + 1] - x[i]) < W / 2:
                draw.line([(x[i], y[i]), (x[i + 1], y[i + 1])], fill=(150, 150, 150), width=1)

    polyline(draw, np.arange(W).astype(float), pred_c, (0, 255, 0))
    polyline(draw, np.arange(W).astype(float), pred_f, (0, 255, 0))
    for c in np.where(np.abs(err_c) > 2)[0]:
        draw.point((int(c), int(true_c[c])), fill=(255, 0, 0))

    txt_order_ok = None
    if corners is not None:
        xs = corners[0::2, 0]
        txt_order_ok = bool(np.all(np.diff(xs) > 0))
        shifted = [(f - int(np.argmin(xs))) % len(xs) for f in fake]
        for k, (x, yc) in enumerate(corners[0::2]):
            yf = corners[2 * k + 1, 1]
            colour = (255, 40, 40) if k in shifted else (255, 230, 0)
            draw.line([(x, yc), (x, yf)], fill=colour, width=2)
            draw.text((x + 3, yc - 12), str(k), fill=colour)
        for k in range(len(xs)):
            a, b = corners[2 * k], corners[(2 * k + 2) % len(corners)]
            if abs(b[0] - a[0]) < W / 2:
                draw.line([tuple(a), tuple(b)], fill=(255, 140, 0), width=1)

    record = {
        "room": room, "view": view,
        "polygon_vertices": len(layout["polygon"]["coordinates"][0]) - 1,
        "collinear_vertices": fake,
        "ceiling_rows_top_on_ceiling": round(ceil_frac_top, 3),
        "ceil_err_px_median": float(np.nanmedian(np.abs(err_c))),
        "ceil_cols_err_gt2px": float(np.nanmean(np.abs(err_c) > 2)),
        "floor_err_px_median": float(np.nanmedian(np.abs(err_f))),
        "floor_cols_err_gt2px": float(np.nanmean(np.abs(err_f) > 2)),
        "floor_cols_seen": float(np.mean(true_f >= 0)),
        "txt_corners": None if corners is None else len(corners) // 2,
        "txt_sorted": txt_order_ok,
        "objects": len(scene["objects"]),
        "objects_checked": len(obj_stats),
        "inside_box_min": min((s["inside_box"] for s in obj_stats), default=None),
        "objects_below_95": [s for s in obj_stats if s["inside_box"] < 0.95],
        "warnings": layout.get("warnings"),
    }
    return img, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("rooms", nargs="+", help="<house>/<room>")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.exp)
    report = []
    for room in args.rooms:
        house, room_id = room.split("/")
        split = splits[room]
        render = json.loads((args.exp / "outputs" / split / room / "render.json").read_text())
        meta = json.loads((args.exp / "ground_truth" / split / room / "0.json").read_text())["metadata"]
        sc = meta["metric_scale"]
        poly = [[x / sc, y / sc] for x, y in meta["layout_geometry_world"]["polygon"]["coordinates"][0]]
        layout = dict(render["layout"], polygon={"type": "Polygon", "coordinates": [poly]})
        fake = collinear_vertices(layout["polygon"])
        tiles = []
        for view in range(4):
            img, record = check_view(args.exp, split, room, view, render, layout, fake)
            tiles.append(img)
            report.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        grid = Image.new("RGB", (2 * W, 2 * H))
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % 2) * W, (i // 2) * H))
        grid.save(args.out / f"{house[:8]}__{room_id}.png")
    (args.out / "report.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in report))


if __name__ == "__main__":
    main()
