from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_requests(path: str) -> list:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "samples" in payload:
        return list(payload["samples"])
    return [payload]


def write_output(
    path: str,
    *,
    layout: "Mapping[str, Any] | None" = None,
    objects: "Iterable[Mapping[str, Any]] | None" = None,
    relations: "Iterable[Mapping[str, Any]] | None" = None,
    metadata: "Mapping[str, Any] | None" = None,
) -> dict:
    payload = {
        "layout": dict(layout) if layout is not None else None,
        "objects": [dict(item) for item in (objects or [])],
        "relations": [dict(item) for item in (relations or [])],
        "metadata": dict(metadata or {}),
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def write_outputs(path: str, outputs) -> dict:
    payload = {"outputs": [dict(item) for item in outputs]}
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def write_scenes(path: str, scenes) -> dict:
    if len(scenes) != 1:
        return write_outputs(path, scenes)
    scene = scenes[0]
    return write_output(
        path,
        layout=scene["layout"],
        objects=scene["objects"],
        relations=scene["relations"],
        metadata=scene["metadata"],
    )


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def seed_everything(seed) -> None:
    if seed is None:
        return
    import random as _random

    _random.seed(seed)
    try:
        import numpy as _numpy

        _numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
    except ImportError:
        pass
