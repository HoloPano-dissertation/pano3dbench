from __future__ import annotations

import contextlib
import json
from collections import ChainMap, UserDict
from pathlib import Path
from typing import Any

import yaml

from benchmark_toolbox.config import ConfigError


# Does it really nead what parameter?
def read_mapping_file(path: Path, *, what: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    elif path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        msg = f"File {path} is not YAML or JSON"
        raise ValueError(msg)
    try:
        return dict(data)
    except TypeError as terr:
        msg = f"{what} must be a mapping: {path}"
        raise ConfigError(msg) from terr


class MapFile(UserDict[str, Any]):
    """Load mapping file and resolve all paths with respect to current file position"""

    path_keys = (
        "manifest",
        "checkpoint",
        "prediction_dir",
        "environment",
    )

    def __init__(self, path: str | Path) -> None:
        path = Path(path).expanduser().resolve()
        map = read_mapping_file(path, what="Map file")
        super().__init__(map)
        self.origin = path
        with contextlib.suppress(KeyError):
            self.data["model"]["parameters"] = {
                k: self._resolve_path(k, v)
                for k, v in self.data["model"]["parameters"].items()
            }
        with contextlib.suppress(KeyError):
            self.data["dataset"]["parameters"] = {
                k: self._resolve_path(k, v)
                for k, v in self.data["dataset"]["parameters"].items()
            }
        with contextlib.suppress(KeyError):
            for metric in self.data["metrics"]:
                metric["parameters"] = {
                    k: self._resolve_path(k, v) for k, v in metric["parameters"].items()
                }

    @classmethod
    def _is_path(cls, key: str) -> bool:
        return key in cls.path_keys or key.endswith(("_path", "_dir", "_root"))

    def _resolve_path(self, key: str, value: Any, /) -> str:  # noqa: ANN401
        if self._is_path(key):
            candidate = self.origin.parent.joinpath(value).expanduser().resolve()
            return str(candidate)
        return value


def load_map_chain(path: str | Path) -> ChainMap[str, Any]:
    """Load map files that can extends each other

    Args:
        path: Path to root mapping file

    Returns:
        Map of all options.
        Parameters that are closer to the root override distant ones

    Raises:
        ConfigError: If cycle import is detected
            (message starts with "Cyclic 'extends'"")

    """
    root_file = MapFile(path)
    maps = [root_file]
    seen = {root_file.origin}
    base = root_file.origin.parent
    last_map = maps[-1]
    while "extends" in last_map:
        extension_path = base.joinpath(last_map.pop("extends")).expanduser().resolve()
        if extension_path in seen:
            chain = " -> ".join(str(item) for item in (*seen, extension_path))
            raise ConfigError(
                f"Cyclic 'extends' in the experiment configuration: {chain}"
            )
        seen.add(extension_path)
        extension = MapFile(extension_path)
        maps.append(extension)
        last_map = maps[-1]
        base = last_map.origin.parent
    return ChainMap(*maps)
