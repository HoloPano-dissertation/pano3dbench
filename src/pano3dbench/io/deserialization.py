from __future__ import annotations

import json
from collections import ChainMap, UserDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from pano3dbench.config import ConfigError


def read_mapping_file(path: Path, *, what: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    elif path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        msg = f"File {path} is not YAML or JSON"
        raise ValueError(msg)
    if not isinstance(data, Mapping):
        msg = f"{what} must be a mapping: {path}"
        raise ConfigError(msg)
    return dict(data)


def _resolve_against(base: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else base.joinpath(candidate).resolve()


class MapFile(UserDict[str, Any]):
    path_keys = (
        "manifest",
        "checkpoint",
        "prediction_dir",
        "environment",
    )

    def __init__(self, path: str | Path) -> None:
        path = Path(path).expanduser().resolve()
        super().__init__(read_mapping_file(path, what="Map file"))
        self.origin = path
        for section in ("model", "dataset"):
            component = self.data.get(section)
            if isinstance(component, dict) and isinstance(component.get("parameters"), dict):
                component["parameters"] = self._resolve_parameters(component["parameters"])
        for metric in self.data.get("metrics") or ():
            if isinstance(metric, dict) and isinstance(metric.get("parameters"), dict):
                metric["parameters"] = self._resolve_parameters(metric["parameters"])

    @classmethod
    def _is_path(cls, key: str) -> bool:
        return key in cls.path_keys or key.endswith(("_path", "_dir", "_root"))

    def _resolve_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return {k: self._resolve_path(k, v) for k, v in parameters.items()}

    def _resolve_path(self, key: str, value: Any, /) -> Any:  # noqa: ANN401
        if self._is_path(key) and isinstance(value, str):
            return str(_resolve_against(self.origin.parent, value))
        return value


def load_map_chain(path: str | Path) -> ChainMap[str, Any]:
    maps = [MapFile(path)]
    seen = [maps[0].origin]
    while "extends" in maps[-1]:
        last_map = maps[-1]
        extension = _resolve_against(last_map.origin.parent, str(last_map.pop("extends"))).resolve()
        if extension in seen:
            chain = " -> ".join(str(item) for item in (*seen, extension))
            raise ConfigError(f"Cyclic 'extends' in the experiment configuration: {chain}")
        seen.append(extension)
        maps.append(MapFile(extension))
    return ChainMap(*maps)
