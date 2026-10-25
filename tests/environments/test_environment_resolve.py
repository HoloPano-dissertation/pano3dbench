from collections.abc import Callable
from pathlib import Path

from pano3dbench.environments import EnvironmentSpec
from pano3dbench.environments.conda import CondaEnvironmentManager
from pano3dbench.environments.venv import VenvEnvironmentManager


def test_conda_resolve_builds_conda_run_prefix(
    load_env: Callable[[str], EnvironmentSpec],
) -> None:
    handle = CondaEnvironmentManager().resolve(load_env("dpc"))

    assert "run" in handle.command_prefix
    assert "Pano3D" in handle.command_prefix
    assert handle.command_prefix[-1] == "python"
    assert handle.env.get("CUDA_VISIBLE_DEVICES") != ""
    handle_path = Path("repos", "Pano3D")
    assert str(handle.cwd).endswith(str(handle_path))
    assert handle.wrapper[0] == "xvfb-run"


def test_venv_resolve_points_at_venv_python(
    load_env: Callable[[str], EnvironmentSpec],
) -> None:
    handle = VenvEnvironmentManager().resolve(load_env("echo"))

    assert len(handle.command_prefix) == 1
    python = handle.command_prefix[0]
    assert python.endswith(
        str(Path("venvs", "echo", "bin", "python"))
    ) or python.endswith(str(Path("venvs", "echo", "Scripts", "python.exe")))
    assert handle.cwd is None
