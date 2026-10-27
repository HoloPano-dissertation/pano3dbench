from __future__ import annotations

from typing import Any, Iterable, Mapping


def box(min_corner: Iterable[float], max_corner: Iterable[float]) -> dict:
    return {
        "min_corner": [float(value) for value in min_corner],
        "max_corner": [float(value) for value in max_corner],
    }


def oriented_box(centroid: Iterable[float], size: Iterable[float], basis) -> dict:
    return {
        "center": [float(value) for value in centroid],
        "size": [float(value) for value in size],
        "basis": [[float(value) for value in row] for row in basis],
    }


def scene_object(
    object_id: str,
    label: str,
    bbox: Mapping[str, Any],
    score: float = 1.0,
    attributes: "Mapping[str, Any] | None" = None,
) -> dict:
    return {
        "object_id": str(object_id),
        "label": str(label),
        "score": float(score),
        "bbox": dict(bbox),
        "attributes": dict(attributes or {}),
    }


def corners_to_axis_aligned(corners) -> dict:
    xs = [float(point[0]) for point in corners]
    ys = [float(point[1]) for point in corners]
    zs = [float(point[2]) for point in corners]
    return box([min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)])


def oriented_box_to_axis_aligned(centroid, basis, coeffs) -> dict:
    centre = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
    corners = []
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                signs = (s0, s1, s2)
                corner = []
                for axis in range(3):
                    value = centre[axis] + sum(
                        signs[k] * float(coeffs[k]) * float(basis[k][axis])
                        for k in range(3)
                    )
                    corner.append(value)
                corners.append(corner)
    return corners_to_axis_aligned(corners)


def scene_from_oriented_objects(objects, layout_corners=None, layout_box=None) -> dict:
    out_objects = []
    for index, item in enumerate(objects):
        size = [2.0 * float(coeff) for coeff in item["coeffs"]]
        bbox = oriented_box(item["centroid"], size, item["basis"])
        attributes = dict(item.get("attributes") or {})
        if item.get("shape") is not None:
            attributes.setdefault("shape", item["shape"])
        out_objects.append(
            scene_object(
                item.get("id", f"obj-{index}"),
                item.get("label", "object"),
                bbox,
                float(item.get("score", 1.0)),
                attributes or None,
            )
        )
    layout = layout_box
    if layout is None and layout_corners is not None:
        layout = corners_to_axis_aligned(layout_corners)
    return {"layout": layout, "objects": out_objects, "relations": []}
