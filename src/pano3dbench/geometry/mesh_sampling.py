from __future__ import annotations

import bisect
import math

from pano3dbench.geometry.point_cloud import Point

Triangle = tuple[int, int, int]


def sample_surface(
    vertices: list[Point], faces: list[Triangle], count: int, rng
) -> list[Point]:
    if not faces:
        return [rng.choice(vertices) for _ in range(count)] if vertices else []
    cumulative: list[float] = []
    total = 0.0
    for i, j, k in faces:
        total += _triangle_area(vertices[i], vertices[j], vertices[k])
        cumulative.append(total)
    if total <= 0.0:
        return [rng.choice(vertices) for _ in range(count)]
    points: list[Point] = []
    last = len(faces) - 1
    for _ in range(count):
        index = bisect.bisect_left(cumulative, rng.random() * total)
        i, j, k = faces[min(index, last)]
        points.append(_barycentric(vertices[i], vertices[j], vertices[k], rng))
    return points


def _triangle_area(a: Point, b: Point, c: Point) -> float:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def _barycentric(a: Point, b: Point, c: Point, rng) -> Point:
    r1 = math.sqrt(rng.random())
    r2 = rng.random()
    wa, wb, wc = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
    return (
        wa * a[0] + wb * b[0] + wc * c[0],
        wa * a[1] + wb * b[1] + wc * c[1],
        wa * a[2] + wb * b[2] + wc * c[2],
    )
