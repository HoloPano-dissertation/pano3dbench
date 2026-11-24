from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pano3dbench.scene import SceneOutput
from pano3dbench.visualization.projection import (
    Panorama,
    Pixel,
    box_polylines,
    label_anchor,
    room_polylines,
)

Color = tuple[int, int, int]
GROUND_TRUTH_COLOR: Color = (40, 200, 80)
PREDICTION_COLOR: Color = (240, 60, 50)


@dataclass(frozen=True)
class Stroke:
    points: tuple[Pixel, ...]
    color: Color
    width: int


@dataclass(frozen=True)
class Label:
    anchor: Pixel
    text: str
    color: Color
    above: bool = True


@dataclass(frozen=True)
class OverlayOptions:
    min_score: float = 0.0
    labels: bool = True
    layout: bool = True
    line_width: int = 2


def overlay_shapes(
    panorama: Panorama,
    ground_truth: Mapping[str, Any] | None,
    prediction: Mapping[str, Any] | None,
    options: OverlayOptions = OverlayOptions(),
) -> tuple[list[Stroke], list[Label]]:
    strokes: list[Stroke] = []
    labels: list[Label] = []
    for document, color, is_prediction in (
        (ground_truth, GROUND_TRUTH_COLOR, False),
        (prediction, PREDICTION_COLOR, True),
    ):
        if document is None:
            continue
        scene = SceneOutput.from_dict(document)
        if options.layout:
            for line in _layout_lines(panorama, scene, document):
                strokes.append(Stroke(tuple(line), color, max(1, options.line_width - 1)))
        for item in scene.objects:
            if is_prediction and item.score < options.min_score:
                continue
            for line in box_polylines(panorama, item.bbox):
                strokes.append(Stroke(tuple(line), color, options.line_width))
            if options.labels:
                text = f"{item.label} {item.score:.2f}" if is_prediction else item.label
                labels.append(
                    Label(label_anchor(panorama, item.bbox), text, color, above=is_prediction)
                )
    return strokes, labels


def _layout_lines(
    panorama: Panorama, scene: SceneOutput, document: Mapping[str, Any]
) -> list[list[Pixel]]:
    if scene.layout is None:
        return []
    room = _room_polygon(document)
    if room is not None:
        corners = scene.layout.as_oriented().corners()
        floor = min(corner[2] for corner in corners)
        ceiling = max(corner[2] for corner in corners)
        return room_polylines(panorama, room, floor, ceiling)
    return box_polylines(panorama, scene.layout)


def _room_polygon(document: Mapping[str, Any]) -> list[tuple[float, float]] | None:
    metadata = document.get("metadata") or {}
    geometry = metadata.get("layout_geometry_world") or {}
    polygon = geometry.get("polygon") or {}
    camera = metadata.get("camera_location_world")
    if polygon.get("type") != "Polygon" or not polygon.get("coordinates") or camera is None:
        return None
    return [
        (float(x) - float(camera[0]), float(y) - float(camera[1]))
        for x, y in polygon["coordinates"][0]
    ]


def render_overlay(
    image_path: str | Path,
    output_path: str | Path,
    ground_truth: Mapping[str, Any] | None = None,
    prediction: Mapping[str, Any] | None = None,
    options: OverlayOptions = OverlayOptions(),
    title: str | None = None,
) -> Path:
    image_module, draw_module, font_module = _pillow()
    with image_module.open(image_path) as source:
        image = source.convert("RGB")
    panorama = Panorama(image.width, image.height)
    strokes, labels = overlay_shapes(panorama, ground_truth, prediction, options)
    canvas = draw_module.Draw(image)
    for stroke in strokes:
        if len(stroke.points) > 1:
            canvas.line(stroke.points, fill=stroke.color, width=stroke.width, joint="curve")
    font = font_module.load_default()
    for label in labels:
        _text(canvas, font, label, image.width)
    _legend(canvas, font, ground_truth is not None, prediction is not None, title)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target


def _text(canvas: Any, font: Any, label: Label, width: int) -> None:  # noqa: ANN401
    left, top, right, bottom = canvas.textbbox((0, 0), label.text, font=font)
    x = min(max(0.0, label.anchor[0] - (right - left) / 2), width - (right - left) - 2)
    y = label.anchor[1] - (bottom - top) - 6 if label.above else label.anchor[1] + 4
    y = max(0.0, y)
    canvas.rectangle((x - 2, y - 1, x + right - left + 2, y + bottom - top + 3), fill=(0, 0, 0))
    canvas.text((x, y), label.text, fill=label.color, font=font)


def _legend(canvas: Any, font: Any, truth: bool, prediction: bool, title: str | None) -> None:  # noqa: ANN401
    entries = []
    if title:
        entries.append((title, (255, 255, 255)))
    if truth:
        entries.append(("ground truth", GROUND_TRUTH_COLOR))
    if prediction:
        entries.append(("prediction", PREDICTION_COLOR))
    y = 4.0
    for text, color in entries:
        _, top, right, bottom = canvas.textbbox((0, 0), text, font=font)
        canvas.rectangle((2, y - 1, 8 + right, y + bottom - top + 3), fill=(0, 0, 0))
        canvas.text((5, y), text, fill=color, font=font)
        y += bottom - top + 6


def _pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError(
            "Drawing needs Pillow: pip install 'pano3dbench[viz]'"
        ) from error
    return Image, ImageDraw, ImageFont


def read_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_overlays(
    manifest: str | Path,
    output_dir: str | Path,
    prediction_dir: str | Path | None = None,
    sample_ids: list[str] | None = None,
    limit: int | None = None,
    options: OverlayOptions = OverlayOptions(),
) -> Iterator[Path]:
    wanted = set(sample_ids) if sample_ids else None
    written = 0
    with Path(manifest).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if wanted is not None and sample_id not in wanted:
                continue
            if limit is not None and written >= limit:
                return
            prediction = None
            if prediction_dir is not None:
                candidate = Path(prediction_dir) / f"{sample_id}.json"
                prediction = read_json(candidate) if candidate.is_file() else None
            yield render_overlay(
                record["input"],
                Path(output_dir) / f"{sample_id}.png",
                read_json(record.get("ground_truth")),
                prediction,
                options,
                title=sample_id,
            )
            written += 1
