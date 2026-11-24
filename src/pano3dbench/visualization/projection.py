from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pano3dbench.geometry.boxes import BoundingBox3D, OrientedBox3D

Point3D = tuple[float, float, float]
Pixel = tuple[float, float]

BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@dataclass(frozen=True)
class Panorama:
    width: int
    height: int

    def project(self, point: Sequence[float]) -> Pixel:
        x, y, z = point
        horizontal = math.hypot(x, y)
        column = ((0.5 + math.atan2(x, y) / (2 * math.pi)) % 1.0) * self.width - 0.5
        row = (0.5 - math.atan2(z, horizontal) / math.pi) * self.height - 0.5
        return column, row

    def polylines(
        self, start: Sequence[float], end: Sequence[float], steps: int = 48
    ) -> list[list[Pixel]]:
        points = [
            self.project(
                tuple(a + (b - a) * index / steps for a, b in zip(start, end))
            )
            for index in range(steps + 1)
        ]
        return self.unwrap(points)

    def unwrap(self, pixels: Sequence[Pixel]) -> list[list[Pixel]]:
        if not pixels:
            return []
        pieces: list[list[Pixel]] = [[pixels[0]]]
        for previous, current in zip(pixels, pixels[1:]):
            jump = current[0] - previous[0]
            if abs(jump) <= self.width / 2:
                pieces[-1].append(current)
                continue
            shift = -self.width if jump > 0 else self.width
            pieces[-1].append((current[0] + shift, current[1]))
            pieces.append([(previous[0] - shift, previous[1]), current])
        return pieces


def box_corners(box: BoundingBox3D | OrientedBox3D) -> list[Point3D]:
    corners = box.as_oriented().corners()
    return [tuple(corner) for corner in corners]


def box_polylines(
    panorama: Panorama, box: BoundingBox3D | OrientedBox3D, steps: int = 48
) -> list[list[Pixel]]:
    corners = box_corners(box)
    lines: list[list[Pixel]] = []
    for first, second in BOX_EDGES:
        lines.extend(panorama.polylines(corners[first], corners[second], steps))
    return lines


def room_polylines(
    panorama: Panorama,
    polygon: Iterable[Sequence[float]],
    floor: float,
    ceiling: float,
    steps: int = 48,
) -> list[list[Pixel]]:
    corners = [(float(x), float(y)) for x, y in polygon]
    if len(corners) > 1 and corners[0] == corners[-1]:
        corners = corners[:-1]
    lines: list[list[Pixel]] = []
    for index, (x, y) in enumerate(corners):
        nx, ny = corners[(index + 1) % len(corners)]
        for height in (floor, ceiling):
            lines.extend(panorama.polylines((x, y, height), (nx, ny, height), steps))
        lines.extend(panorama.polylines((x, y, floor), (x, y, ceiling), steps))
    return lines


def label_anchor(panorama: Panorama, box: BoundingBox3D | OrientedBox3D) -> Pixel:
    oriented = box.as_oriented()
    top = max(corner[2] for corner in oriented.corners())
    return panorama.project((oriented.center[0], oriented.center[1], top))
