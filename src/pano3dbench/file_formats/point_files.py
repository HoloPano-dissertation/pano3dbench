from __future__ import annotations

import array
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pano3dbench.geometry.point_cloud import Point


def as_points(values: Any) -> list[Point]:
    points: list[Point] = []
    for item in values:
        if len(item) < 3:
            raise ValueError("each point needs three coordinates")
        points.append((float(item[0]), float(item[1]), float(item[2])))
    return points


def write_f32_points(path: str | Path, points: Sequence[Point]) -> None:
    flat = array.array("f", (coordinate for point in points for coordinate in point))
    if sys.byteorder != "little":
        flat.byteswap()
    Path(path).write_bytes(flat.tobytes())


def read_f32_points(path: Path) -> list[Point]:
    flat = array.array("f")
    data = path.read_bytes()
    flat.frombytes(data[: len(data) - len(data) % 4])
    if sys.byteorder != "little":
        flat.byteswap()
    return [
        (flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat) - 2, 3)
    ]
