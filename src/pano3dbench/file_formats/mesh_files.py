from __future__ import annotations

import struct
from pathlib import Path

from pano3dbench.geometry.mesh_sampling import Triangle
from pano3dbench.geometry.point_cloud import Point


_PLY_TYPE_TO_STRUCT = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def read_ply(path: Path) -> tuple[list[Point], list[Triangle]]:
    raw = path.read_bytes()
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise ValueError(f"{path}: PLY without end_header")
    line_end = raw.find(b"\n", header_end)
    header = raw[:line_end].decode("ascii", "replace").splitlines()
    body = raw[line_end + 1 :]

    fmt = "ascii"
    elements: list[tuple[str, int, list[tuple[str, str, str]]]] = []
    for line in header:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "format":
            fmt = tokens[1]
        elif tokens[0] == "element":
            elements.append((tokens[1], int(tokens[2]), []))
        elif tokens[0] == "property" and elements:
            if tokens[1] == "list":
                elements[-1][2].append(("list", tokens[2], tokens[3]))
            else:
                elements[-1][2].append(("scalar", tokens[1], tokens[2]))

    if fmt == "ascii":
        return _read_ply_ascii(body.decode("ascii", "replace"), elements)
    if fmt == "binary_little_endian":
        return _read_ply_binary(body, elements)
    raise ValueError(f"{path}: unsupported PLY format '{fmt}'")


def _read_ply_ascii(body: str, elements) -> tuple[list[Point], list[Triangle]]:
    tokens = body.split()
    cursor = 0
    vertices: list[Point] = []
    faces: list[Triangle] = []
    for name, count, properties in elements:
        if name == "vertex":
            axis = {prop[2]: index for index, prop in enumerate(properties)}
            width = len(properties)
            xi, yi, zi = axis["x"], axis["y"], axis["z"]
            for _ in range(count):
                row = tokens[cursor : cursor + width]
                cursor += width
                vertices.append((float(row[xi]), float(row[yi]), float(row[zi])))
        elif name == "face":
            for _ in range(count):
                degree = int(tokens[cursor])
                indices = [int(tokens[cursor + 1 + k]) for k in range(degree)]
                cursor += 1 + degree
                _fan(indices, faces)
        else:
            width = len(properties)
            cursor += count * width
    return vertices, faces


def _read_ply_binary(body: bytes, elements) -> tuple[list[Point], list[Triangle]]:
    vertices: list[Point] = []
    faces: list[Triangle] = []
    offset = 0
    for name, count, properties in elements:
        if name == "vertex":
            fmt = "<" + "".join(
                _PLY_TYPE_TO_STRUCT[prop[1]] for prop in properties
            )
            size = struct.calcsize(fmt)
            unpack = struct.Struct(fmt).unpack_from
            axis = {prop[2]: index for index, prop in enumerate(properties)}
            xi, yi, zi = axis["x"], axis["y"], axis["z"]
            for _ in range(count):
                row = unpack(body, offset)
                offset += size
                vertices.append((float(row[xi]), float(row[yi]), float(row[zi])))
        elif name == "face":
            list_prop = next(p for p in properties if p[0] == "list")
            count_char = _PLY_TYPE_TO_STRUCT[list_prop[1]]
            index_char = _PLY_TYPE_TO_STRUCT[list_prop[2]]
            count_size = struct.calcsize(count_char)
            index_size = struct.calcsize(index_char)
            for _ in range(count):
                degree = struct.unpack_from("<" + count_char, body, offset)[0]
                offset += count_size
                indices = [
                    struct.unpack_from("<" + index_char, body, offset + k * index_size)[0]
                    for k in range(degree)
                ]
                offset += degree * index_size
                _fan(indices, faces)
        else:
            fmt = "<" + "".join(
                _PLY_TYPE_TO_STRUCT[prop[1]] for prop in properties if prop[0] == "scalar"
            )
            offset += count * struct.calcsize(fmt)
    return vertices, faces


def read_obj(path: Path) -> tuple[list[Point], list[Triangle]]:
    vertices: list[Point] = []
    faces: list[Triangle] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            _, x, y, z, *_ = line.split()
            vertices.append((float(x), float(y), float(z)))
        elif line.startswith("f "):
            indices = [
                int(token.split("/", 1)[0]) - 1
                for token in line.split()[1:]
            ]
            _fan(indices, faces)
    return vertices, faces


def _fan(indices: list[int], faces: list[Triangle]) -> None:
    for k in range(1, len(indices) - 1):
        faces.append((indices[0], indices[k], indices[k + 1]))
