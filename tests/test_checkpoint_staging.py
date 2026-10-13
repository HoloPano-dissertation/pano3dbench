from pathlib import Path
from urllib.error import URLError

import pytest

from benchmark_toolbox.environments.spec import CheckpointSpec
from benchmark_toolbox.environments.steps import download_checkpoint

NEVER_FETCHED = "https://invalid.example/never-fetched"


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(artifacts_dir, repo_dir) with ``model.pth`` staged and
    the env var pointing at it."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model.pth").write_bytes(b"weights")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("BENCHMARK_TOOLBOX_ARTIFACTS", str(artifacts))
    return artifacts, repo


def test_uses_local_artifact_instead_of_url(staged: tuple[Path, Path]) -> None:
    _, repo = staged
    spec = CheckpointSpec(url=NEVER_FETCHED, dest="ckpt/model.pth")
    assert download_checkpoint(spec, repo).read_bytes() == b"weights"


def test_local_artifact_checksum_is_verified(staged: tuple[Path, Path]) -> None:
    _, repo = staged
    spec = CheckpointSpec(url=NEVER_FETCHED, dest="ckpt/model.pth", sha256="0" * 64)
    with pytest.raises(RuntimeError):
        download_checkpoint(spec, repo)


def test_no_artifacts_dir_means_no_local_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BENCHMARK_TOOLBOX_ARTIFACTS", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    spec = CheckpointSpec(url=NEVER_FETCHED, dest="ckpt/model.pth")
    with pytest.raises(URLError):
        download_checkpoint(spec, repo)
