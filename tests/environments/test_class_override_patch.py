from pathlib import Path

import pytest
import yaml

from pano3dbench.environments.patches import apply_patches

ORIGINAL = """import os

IG56CLASSES = ['basket', 'bathtub', 'bed']

IG59CLASSES = IG56CLASSES + ['walls', 'floors', 'ceilings']
"""


@pytest.fixture
def class_patch(repo_root: Path) -> list[dict]:
    spec = yaml.safe_load((repo_root / "configs/environments/dpc.yaml").read_text())
    return [p for p in spec["patches"] if p.get("file") == "configs/data_config.py"]


@pytest.fixture
def prepared(tmp_path: Path, class_patch: list[dict]) -> Path:
    target = tmp_path / "configs" / "data_config.py"
    target.parent.mkdir(parents=True)
    target.write_text(ORIGINAL, encoding="utf-8")
    apply_patches(class_patch, tmp_path)
    return target


def run(target: Path) -> dict:
    namespace = {}
    exec(compile(target.read_text(), str(target), "exec"), namespace)
    return namespace


def test_the_patch_is_declared_for_the_class_list(class_patch: list[dict]) -> None:
    assert len(class_patch) == 1


def test_without_the_variable_the_igibson_classes_are_kept(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PANO3D_CLASSES", raising=False)
    namespace = run(prepared)
    assert namespace["IG56CLASSES"] == ["basket", "bathtub", "bed"]
    assert namespace["IG59CLASSES"][-3:] == ["walls", "floors", "ceilings"]


def test_with_the_variable_our_classes_replace_them(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PANO3D_CLASSES", "bed,chair,table")
    namespace = run(prepared)
    assert namespace["IG56CLASSES"] == ["bed", "chair", "table"]
    assert namespace["IG59CLASSES"] == [
        "bed",
        "chair",
        "table",
        "walls",
        "floors",
        "ceilings",
    ]


def test_the_patch_does_not_repeat_a_line_that_already_exists(prepared: Path) -> None:
    assert "PANO3D_CLASSES" in prepared.read_text(), (
        "the guard on repeated lines must not skip it"
    )


def test_applying_the_patch_twice_changes_nothing(
    prepared: Path, tmp_path: Path, class_patch: list[dict]
) -> None:
    once = prepared.read_text()
    apply_patches(class_patch, tmp_path)
    assert prepared.read_text() == once
