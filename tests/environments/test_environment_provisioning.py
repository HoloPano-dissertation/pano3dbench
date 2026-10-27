from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import pytest

from pano3dbench.environments import EnvironmentSpec
from pano3dbench.environments.base import EnvHandle, EnvironmentManager
from pano3dbench.environments.plan import (
    DONE,
    NETWORK,
    TODO,
    build_plan,
    render_plan,
    verify_checkpoints,
)
from pano3dbench.environments.spec import CheckpointSpec, load_environment_spec
from pano3dbench.environments.steps import checkpoint_path, download_checkpoint

REPO = Path("/repo")


@pytest.fixture
def set_env(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:

    def _set(**values: "str | None") -> None:
        for key, value in values.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)

    return _set


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    path = tmp_path / "staged"
    path.mkdir()
    return path


class _FakeManager(EnvironmentManager):
    def __init__(self, exists: bool = False) -> None:
        super().__init__()
        self.exists = exists

    def prepare(self, spec) -> NoReturn:  # noqa: ANN001, ARG002
        raise AssertionError("prepare must not run in these tests")

    def resolve(self, spec) -> EnvHandle:  # noqa: ANN001, ARG002
        return EnvHandle()

    def environment_exists(self, spec) -> bool:  # noqa: ANN001, ARG002
        return self.exists


def _spec(tmp: Path, **overrides: Any) -> EnvironmentSpec:  # noqa: ANN401
    data = {
        "name": "demo",
        "backend": "venv",
        "workspace": str(tmp / "workspace"),
        "runner": {"entry": "runners/echo_runner.py"},
    }
    data.update(overrides)
    return load_environment_spec(data, base_dir=tmp)


def test_relative_destination_lands_in_the_repository() -> None:
    checkpoint = CheckpointSpec(url="https://example/x", dest="out/model.pth")
    assert checkpoint_path(checkpoint, REPO) == Path("/repo/out/model.pth")


def test_home_destination_is_expanded() -> None:
    backbone = CheckpointSpec(
        url="https://example/resnet.pth", dest="~/.cache/torch/hub/checkpoints/r.pth"
    )
    assert (
        checkpoint_path(backbone, REPO)
        == Path.home() / ".cache/torch/hub/checkpoints/r.pth"
    )


def test_absolute_destination_is_taken_literally() -> None:
    absolute = CheckpointSpec(url="https://example/x", dest="/opt/weights/m.pth")
    assert checkpoint_path(absolute, REPO) == Path("/opt/weights/m.pth")


def test_staged_file_reaches_a_destination_outside_the_repository(
    tmp_path: Path, staging: Path, set_env: Callable[..., None]
) -> None:
    (staging / "r.pth").write_bytes(b"backbone")
    target = tmp_path / "cache" / "r.pth"
    checkpoint = CheckpointSpec(
        url="https://invalid.example/never-fetched", dest=str(target)
    )

    set_env(PANO3DBENCH_ARTIFACTS=str(staging))
    download_checkpoint(checkpoint, tmp_path / "repo")

    assert target.read_bytes() == b"backbone"


def test_missing_checkpoint_fails_fast_and_says_what_to_stage(
    tmp_path: Path, set_env: Callable[..., None]
) -> None:
    checkpoint = CheckpointSpec(
        url="https://example.invalid/weights.pth", dest="ckpt/weights.pth"
    )
    set_env(PANO3DBENCH_OFFLINE="1", PANO3DBENCH_ARTIFACTS=None)

    with pytest.raises(RuntimeError) as caught:
        download_checkpoint(checkpoint, tmp_path)

    message = str(caught.value)
    assert "Offline mode" in message
    assert "https://example.invalid/weights.pth" in message
    assert "weights.pth" in message


def test_staged_checkpoint_still_works_offline(
    tmp_path: Path, staging: Path, set_env: Callable[..., None]
) -> None:
    (staging / "weights.pth").write_bytes(b"w")
    checkpoint = CheckpointSpec(
        url="https://example.invalid/weights.pth", dest="ckpt/weights.pth"
    )
    set_env(PANO3DBENCH_OFFLINE="1", PANO3DBENCH_ARTIFACTS=str(staging))

    assert download_checkpoint(checkpoint, tmp_path).read_bytes() == b"w"


def test_plan_separates_what_is_done_from_what_needs_the_network(
    tmp_path: Path, staging: Path, set_env: Callable[..., None]
) -> None:
    (staging / "staged.pth").write_bytes(b"w")
    spec = _spec(
        tmp_path,
        repo={"url": "https://example/model.git"},
        pip=["torch==1.7.1"],
        checkpoints=[
            {"url": "https://example/a.pth", "dest": "ckpt/staged.pth"},
            {"url": "https://example/b.pth", "dest": "ckpt/missing.pth"},
        ],
        system=["xvfb"],
    )
    set_env(PANO3DBENCH_ARTIFACTS=str(staging))

    steps = build_plan(spec, _FakeManager(exists=True))
    report = render_plan(spec, steps)

    by_summary = {step.summary: step for step in steps}
    assert by_summary["venv environment 'demo' exists"].state == DONE
    assert by_summary["clone repository"].state == NETWORK
    assert by_summary["checkpoint ckpt/staged.pth"].state == TODO
    assert by_summary["checkpoint ckpt/missing.pth"].state == NETWORK
    assert "step(s) will use the network" in report
    assert "xvfb" in report


def test_plan_reports_a_fully_staged_build_as_offline_capable(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    report = render_plan(spec, build_plan(spec, _FakeManager(exists=True)))
    assert "Nothing left to fetch" in report


def test_verify_flags_a_corrupted_staged_file(tmp_path: Path) -> None:
    manager = _FakeManager(exists=True)
    spec = _spec(
        tmp_path,
        repo={"url": "https://example/model.git"},
        checkpoints=[
            {
                "url": "https://example/a.pth",
                "dest": "ckpt/model.pth",
                "sha256": "0" * 64,
            }
        ],
    )
    target = manager.repo_dir(spec) / "ckpt" / "model.pth"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"<html>Sign in</html>")

    problems = verify_checkpoints(spec, manager)

    assert len(problems) == 1
    assert "sha256" in problems[0]


def test_adopt_marks_an_externally_built_environment_as_prepared(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    manager = _FakeManager(exists=True)
    assert not manager.is_prepared(spec)
    manager.adopt(spec)
    assert manager.is_prepared(spec)


def test_adopt_refuses_when_the_environment_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        _FakeManager(exists=False).adopt(_spec(tmp_path))
