"""Shared fixtures for the benchmark_toolbox test suite."""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from benchmark_toolbox.domain import SceneSample
from benchmark_toolbox.environments import EnvironmentSpec, load_environment_spec


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fake_runner() -> Path:
    return Path(__file__).parent / "fixtures" / "fake_model_runner.py"


@pytest.fixture
def fake_command(fake_runner: Path) -> list[str]:
    return [
        sys.executable,
        str(fake_runner),
        "--request={request}",
        "--output={output}",
    ]


@pytest.fixture
def load_env(repo_root: Path) -> Callable[[str], EnvironmentSpec]:
    """Load a shipped environment spec by short name (``"dpc"``, ``"echo"``...)."""

    def _load(name: str) -> EnvironmentSpec:
        return load_environment_spec(
            repo_root / "configs" / "environments" / f"{name}.yaml"
        )

    return _load


@pytest.fixture
def make_samples() -> Callable[[int], list[SceneSample]]:
    def _make(count: int) -> list[SceneSample]:
        return [
            SceneSample(sample_id=f"scene-{index:03d}", input_path=Path(__file__))
            for index in range(count)
        ]

    return _make
