from collections.abc import Callable
from pathlib import Path

import pytest

from pano3dbench.environments import EnvironmentSpec, load_environment_spec


def test_load_echo_spec_from_file(load_env: Callable[[str], EnvironmentSpec]) -> None:
    spec = load_env("echo")
    assert spec.name == "echo"
    assert spec.backend == "venv"
    assert spec.runner.entry == "runners/echo_runner.py"
    assert spec.variant == "cpu"
    assert spec.repo is None


def test_load_dpc_spec_real_fields(load_env: Callable[[str], EnvironmentSpec]) -> None:
    spec = load_env("dpc")
    assert spec.backend == "conda"
    assert spec.name == "Pano3D"
    assert spec.repo is not None
    assert "DeepPanoContext" in spec.repo.url
    assert spec.checkpoints
    assert spec.runner.wrapper[0] == "xvfb-run"
    assert spec.runner.args[:2] == ("--config", "configs/pano3d_igibson.yaml")


def test_inline_dict_requires_runner_entry(repo_root: Path) -> None:
    with pytest.raises(ValueError, match="Environment spec requires"):
        load_environment_spec({"name": "x", "backend": "venv"}, base_dir=repo_root)


def test_fingerprint_is_stable_and_variant_sensitive(repo_root: Path) -> None:
    base = {
        "name": "x",
        "backend": "conda",
        "python": "3.8",
        "runner": {"entry": "r.py"},
    }

    def load(variant: str) -> EnvironmentSpec:
        return load_environment_spec({**base, "variant": variant}, base_dir=repo_root)

    assert load("cpu").fingerprint() == load("cpu").fingerprint()
    assert load("cpu").fingerprint() != load("gpu").fingerprint()
