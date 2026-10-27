from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

from pano3dbench.config.loader import ConfigError, load_map_chain


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
        return cls(type, map.get("parameters") or {})


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
        output_dir = Path(str(layered.get("output_dir") or "artifacts")).expanduser()
        if not output_dir.is_absolute():
            output_dir = source_path.parent.joinpath(output_dir).resolve()
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
        metrics = chain.from_iterable(
            _expand_classes(metric)
            if (metric.get("parameters") or {}).get("per_class")
            else (ComponentConfig.fromdict(metric),)
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


def _expand_classes(metric: Mapping[str, Any]) -> Iterator[ComponentConfig]:
    parameters = dict(metric["parameters"])
    parameters.pop("per_class")
    classes = parameters.get("classes")
    if not classes:
        raise ConfigError(
            f"A 'per_class' metric requires a non-empty 'classes' list: {metric!r}"
        )
    prefix = str(parameters.pop("name", "ap"))
    for class_name in classes:
        yield ComponentConfig.fromdict(
            {
                **metric,
                "parameters": {
                    **parameters,
                    "classes": [class_name],
                    "name": f"{prefix}_{class_name}",
                },
            }
        )
