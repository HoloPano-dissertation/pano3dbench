from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

from benchmark_toolbox.io.deserialization import load_map_chain


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
                "Component configuration requires a 'type' field"
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
        source_path = Path(path).expanduser().resolve()
        layered = load_map_chain(source_path)
        output_dir = Path(layered.get("output_dir", "artifacts")).expanduser().resolve()
        metrics_data = layered.get("metrics", [])
        if not metrics_data:
            raise ConfigError("At least one metric must be configured")
        try:
            model = ComponentConfig.fromdict(layered["model"])
            dataset = ComponentConfig.fromdict(layered["dataset"])
        except KeyError as kerr:
            missing_key = kerr.args[0]
            raise ConfigError(
                f"There is no {missing_key} in presented config {path}"
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
    classes = parameters.get("classes")
    if not classes:
        raise ValueError("A 'per_class' metric requires a non-empty 'classes' list")
    prefix = str(parameters.pop("name", "ap"))
    for class_name in classes:
        metric["parameters"] = parameters | {
            "classes": [class_name],
            "name": f"{prefix}_{class_name}",
        }
        yield ComponentConfig.fromdict(metric)
