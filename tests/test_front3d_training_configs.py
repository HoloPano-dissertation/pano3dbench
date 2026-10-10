from pathlib import Path
from typing import Any

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "dpc"
CONFIGS = sorted(CONFIG_DIR.glob("*front3d*.yaml"))
EXPECTED = {
    "bdb3d_estimation_front3d.yaml",
    "ldif_front3d.yaml",
    "mgnet_front3d.yaml",
    "relation_scene_gcn_front3d.yaml",
}


@pytest.fixture(scope="module")
def configs() -> dict[str, Any]:
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in CONFIGS
    }


def test_every_component_has_a_configuration() -> None:
    assert {path.name for path in CONFIGS} == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_no_path_still_points_at_igibson_data(
    configs: dict[str, Any], name: str
) -> None:
    config = configs[name]
    for key, value in config.get("data", {}).items():
        if isinstance(value, str) and "/" in value:
            assert "igibson" not in value, f"{name}: {key} = {value}"
    for weight in config.get("weight", []):
        assert "igibson" not in weight, name


@pytest.mark.parametrize(
    ("name", "suffix"),
    [
        ("bdb3d_estimation_front3d.yaml", "dpc_scenes"),
        ("relation_scene_gcn_front3d.yaml", "dpc_scenes"),
        ("ldif_front3d.yaml", "objects"),
        ("mgnet_front3d.yaml", "objects"),
    ],
)
def test_scene_and_object_datasets_are_the_expected_ones(
    configs: dict[str, Any], name: str, suffix: str
) -> None:
    assert configs[name]["data"]["split"].endswith(suffix)


def test_box_estimation_starts_from_scratch(configs: dict[str, Any]) -> None:
    config = configs["bdb3d_estimation_front3d.yaml"]
    assert config["weight"] == []
    assert config["finetune"] is False


def test_the_relation_stage_declares_its_two_inputs(configs: dict[str, Any]) -> None:
    weights = configs["relation_scene_gcn_front3d.yaml"]["weight"]
    assert len(weights) == 2
    assert any("bdb3d_estimation_front3d" in w for w in weights)
    assert any("ldif_front3d" in w for w in weights)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_log_paths_do_not_collide_with_the_igibson_runs(
    configs: dict[str, Any], name: str
) -> None:
    path = configs[name].get("log", {}).get("path", "")
    assert path.endswith("front3d"), name
