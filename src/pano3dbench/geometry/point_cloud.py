from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float, float]


class _KDNode:
    __slots__ = ("axis", "point", "left", "right")

    def __init__(self, axis: int, point: Point, left, right) -> None:
        self.axis = axis
        self.point = point
        self.left = left
        self.right = right


def build_kd_tree(points: list[Point], depth: int = 0):
    if not points:
        return None
    axis = depth % 3
    points.sort(key=lambda p: p[axis])
    mid = len(points) // 2
    return _KDNode(
        axis,
        points[mid],
        build_kd_tree(points[:mid], depth + 1),
        build_kd_tree(points[mid + 1 :], depth + 1),
    )


def nearest_squared_distance(root, query: Point) -> float:
    best = math.inf
    stack = [root]
    qx, qy, qz = query
    while stack:
        node = stack.pop()
        if node is None:
            continue
        px, py, pz = node.point
        dx, dy, dz = qx - px, qy - py, qz - pz
        distance = dx * dx + dy * dy + dz * dz
        if distance < best:
            best = distance
        diff = query[node.axis] - node.point[node.axis]
        near = node.left if diff < 0 else node.right
        far = node.right if diff < 0 else node.left
        if diff * diff < best:
            stack.append(far)
        stack.append(near)
    return best


def chamfer_and_fscore(
    pred_points: Sequence[Point],
    gt_points: Sequence[Point],
    *,
    squared: bool = True,
    fscore_threshold: float = 0.1,
) -> tuple[float, float]:
    if not pred_points or not gt_points:
        return math.nan, math.nan
    gt_tree = build_kd_tree(list(gt_points))
    pred_tree = build_kd_tree(list(pred_points))
    pred_to_gt = [nearest_squared_distance(gt_tree, point) for point in pred_points]
    gt_to_pred = [nearest_squared_distance(pred_tree, point) for point in gt_points]

    if squared:
        chamfer = (
            sum(pred_to_gt) / len(pred_to_gt) + sum(gt_to_pred) / len(gt_to_pred)
        )
    else:
        chamfer = sum(math.sqrt(d) for d in pred_to_gt) / len(pred_to_gt) + sum(
            math.sqrt(d) for d in gt_to_pred
        ) / len(gt_to_pred)

    tau_sq = fscore_threshold * fscore_threshold
    precision = sum(1 for d in pred_to_gt if d <= tau_sq) / len(pred_to_gt)
    recall = sum(1 for d in gt_to_pred if d <= tau_sq) / len(gt_to_pred)
    fscore = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return chamfer, fscore


def canonicalize(points: list[Point]) -> list[Point]:
    lo = [min(p[a] for p in points) for a in range(3)]
    hi = [max(p[a] for p in points) for a in range(3)]
    centre = [(lo[a] + hi[a]) / 2.0 for a in range(3)]
    diag = math.sqrt(sum((hi[a] - lo[a]) ** 2 for a in range(3))) or 1.0
    return [
        ((p[0] - centre[0]) / diag, (p[1] - centre[1]) / diag, (p[2] - centre[2]) / diag)
        for p in points
    ]


def normalize_pair(
    pred: list[Point], gt: list[Point], mode: str
) -> tuple[list[Point], list[Point]]:
    if mode == "none":
        return pred, gt
    if mode == "canonical":
        return canonicalize(pred), canonicalize(gt)
    if mode != "unit":
        raise ValueError(f"unknown normalize mode '{mode}'")
    cx = sum(p[0] for p in gt) / len(gt)
    cy = sum(p[1] for p in gt) / len(gt)
    cz = sum(p[2] for p in gt) / len(gt)
    lo = [min(p[a] for p in gt) for a in range(3)]
    hi = [max(p[a] for p in gt) for a in range(3)]
    diag = math.sqrt(sum((hi[a] - lo[a]) ** 2 for a in range(3))) or 1.0

    def apply(points: list[Point]) -> list[Point]:
        return [
            ((p[0] - cx) / diag, (p[1] - cy) / diag, (p[2] - cz) / diag)
            for p in points
        ]

    return apply(pred), apply(gt)
