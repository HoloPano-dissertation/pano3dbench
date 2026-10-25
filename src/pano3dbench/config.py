from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class ComponentConfig:
    type: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fromdict(cls, map: Mapping[str, Any]) -> ComponentConfig:
        try:
            type = str(map["type"])
        except KeyError as kerr:
            raise ConfigError(
                f"Component configuration requires a 'type' field: {map!r}"
            ) from kerr
        return cls(type, map.get("parameters", {}))


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    output_dir: Path
    model: ComponentConfig
    dataset: ComponentConfig
    metrics: tuple[ComponentConfig, ...]
    source_path: Path

    @classmethod
    def fromfile(cls, path: str | Path) -> ExperimentConfig:
        from pano3dbench.io.deserialization import load_map_chain

        source_path = Path(path).expanduser().resolve()
        layered = load_map_chain(source_path)
        if odir := layered.get("output_dir"):
            output_dir = Path(odir).expanduser()
            if not output_dir.is_absolute():
                output_dir = source_path.parent.joinpath(output_dir)
        else:
            output_dir = source_path.parent.joinpath("artifacts")
        output_dir = output_dir.resolve()
        metrics_data = layered.get("metrics", [])
        if not metrics_data:
            raise ConfigError("At least one metric must be configured")
        try:
            model = ComponentConfig.fromdict(layered["model"])
            dataset = ComponentConfig.fromdict(layered["dataset"])
        except KeyError as kerr:
            missing_key = kerr.args[0]
            raise ConfigError(
                f"There is no {missing_key} in presented config: {layered!r}"
            ) from None
        # per_class is a blunt marker. it never appears in single class configs
        # TODO(Daniel): Make another condition of multiclass metrics
        metrics = chain.from_iterable(
            (ComponentConfig.fromdict(metric),)
            if "per_class" not in metric.get("parameters", {})
            else _expand_classes(metric)
            for metric in metrics_data
        )
        return cls(
            str(layered.get("experiment_name", source_path.stem)),
            int(layered.get("seed", 42)),
            output_dir,
            model,
            dataset,
            tuple(metrics),
            source_path,
        )


def _expand_classes(metric: MutableMapping[str, Any]) -> Iterator[ComponentConfig]:
    parameters = metric.get("parameters", {})
    with contextlib.suppress(KeyError):
        parameters.pop("per_class")
    classes = parameters.get("classes")
    if not classes:
        raise ConfigError(
            f"A 'per_class' metric requires a non-empty 'classes' list: {metric!r}"
        )
    prefix = str(parameters.pop("name", "ap"))
    for class_name in classes:
        metric["parameters"] = parameters | {
            "classes": [class_name],
            "name": f"{prefix}_{class_name}",
        }
        yield ComponentConfig.fromdict(metric)
