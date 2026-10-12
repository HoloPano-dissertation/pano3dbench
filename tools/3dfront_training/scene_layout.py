from __future__ import annotations

import numpy as np
from shapely.geometry import shape
from shapely.geometry.polygon import orient


def manhattan_world(layout):
    polygon = orient(shape(layout["polygon"]), sign=-1.0)
    ring = np.asarray(polygon.exterior.coords, dtype=float)[:-1, :2]
    floor = np.column_stack((ring, np.full(len(ring), float(layout["floor_z"]))))
    ceiling = np.column_stack((ring, np.full(len(ring), float(layout["ceiling_z"]))))
    return np.concatenate((floor, ceiling))


def dpc_bdb3d(box, camera_position):
    return {
        "centroid": np.asarray(box["center"], dtype=float)
        + np.asarray(camera_position, dtype=float),
        "basis": np.asarray(box["basis"], dtype=float).T,
        "size": np.asarray(box["size"], dtype=float),
    }


def share_one_column(corners_pix):
    corners_pix = np.array(corners_pix, dtype=float)
    corners_pix[1::2, 0] = corners_pix[0::2, 0]
    return corners_pix


def equirectangular_pixels(points, camera_position, width, height):
    delta = np.asarray(points, dtype=float) - np.asarray(camera_position, dtype=float)
    distance = np.hypot(delta[:, 0], delta[:, 1])
    x = ((0.5 + np.arctan2(delta[:, 0], delta[:, 1]) / (2 * np.pi)) % 1.0) * width - 0.5
    y = (0.5 - np.arctan2(delta[:, 2], distance) / np.pi) * height - 0.5
    return np.column_stack((x, y))


def manhattan_pix(corners_world, camera_position, width, height):
    corners_world = np.asarray(corners_world, dtype=float)
    count = len(corners_world) // 2
    pixels = np.empty((2 * count, 2), dtype=float)
    pixels[0::2] = equirectangular_pixels(corners_world[count:], camera_position, width, height)
    pixels[1::2] = equirectangular_pixels(corners_world[:count], camera_position, width, height)
    pixels = share_one_column(pixels)
    return np.roll(pixels, -2 * int(np.argmin(pixels[0::2, 0])), axis=0)
