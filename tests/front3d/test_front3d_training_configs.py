from pathlib import Path
from typing import Any

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "front3d" / "dpc"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
TRAINING = {
    "first_stage_front3d.yaml",
    "bdb3d_estimation_front3d.yaml",
    "ldif_front3d.yaml",
    "mgnet_front3d.yaml",
    "relation_scene_gcn_front3d.yaml",
}
INFERENCE = {
    "pano3d_front3d.yaml",
    "pano3d_front3d_s20.yaml",
    "pano3d_front3d_im3d.yaml",
    "pano3d_front3d_total3d.yaml",
    "shape500_im3d.yaml",
    "shape500_total3d.yaml",
}


@pytest.fixture(scope="module")
def configs() -> dict[str, Any]:
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in CONFIGS
    }


def test_every_component_has_a_configuration() -> None:
    assert {path.name for path in CONFIGS} == TRAINING | INFERENCE


@pytest.mark.parametrize("name", sorted(TRAINING | INFERENCE))
def test_no_path_still_points_at_igibson_data(
    configs: dict[str, Any], name: str
) -> None:
    config = configs[name]
    for key, value in config.get("data", {}).items():
        if isinstance(value, str) and "/" in value:
            assert "igibson" not in value, f"{name}: {key} = {value}"
    for weight in config.get("weight", []):
        assert "igibson" not in weight, name


@pytest.mark.parametrize("name", sorted(TRAINING | INFERENCE))
def test_paths_are_placeholders(name: str) -> None:
    text = (CONFIG_DIR / name).read_text(encoding="utf-8")
    for line in text.splitlines():
        if "'/" in line or ": /" in line:
            assert "/PATH/TO/" in line, f"{name}: {line.strip()}"


@pytest.mark.parametrize(
    ("name", "suffix"),
    [
        ("first_stage_front3d.yaml", "dpc_scenes"),
        ("bdb3d_estimation_front3d.yaml", "dpc_stage1"),
        ("relation_scene_gcn_front3d.yaml", "dpc_stage1"),
        ("ldif_front3d.yaml", "objects"),
        ("mgnet_front3d.yaml", "objects"),
        ("pano3d_front3d.yaml", "dpc_scenes/test_split.json"),
        ("pano3d_front3d_s20.yaml", "dpc_scenes/test_split.json"),
        ("pano3d_front3d_im3d.yaml", "dpc_scenes/test_split.json"),
        ("pano3d_front3d_total3d.yaml", "dpc_scenes/test_split.json"),
        ("shape500_im3d.yaml", "dpc_scenes/test_split_shape500.json"),
        ("shape500_total3d.yaml", "dpc_scenes/test_split_shape500.json"),
    ],
)
def test_scene_and_object_datasets_are_the_expected_ones(
    configs: dict[str, Any], name: str, suffix: str
) -> None:
    assert configs[name]["data"]["split"].endswith(suffix)


def test_the_first_stage_saves_the_scenes_box_estimation_reads(configs: dict[str, Any]) -> None:
    saved = configs["first_stage_front3d.yaml"]["log"]["save_as_dataset"]
    assert saved == configs["bdb3d_estimation_front3d.yaml"]["data"]["split"]
    assert saved == configs["relation_scene_gcn_front3d.yaml"]["data"]["split"]


def test_box_estimation_starts_from_scratch(configs: dict[str, Any]) -> None:
    config = configs["bdb3d_estimation_front3d.yaml"]
    assert config["weight"] == []
    assert config["finetune"] is False


def test_the_relation_stage_declares_its_two_inputs(configs: dict[str, Any]) -> None:
    weights = configs["relation_scene_gcn_front3d.yaml"]["weight"]
    assert len(weights) == 2
    assert any("bdb3d_estimation_front3d" in w for w in weights)
    assert any("ldif_front3d" in w for w in weights)


@pytest.mark.parametrize("name", sorted(TRAINING))
def test_log_paths_do_not_collide_with_the_igibson_runs(
    configs: dict[str, Any], name: str
) -> None:
    path = configs[name].get("log", {}).get("path", "")
    assert path.endswith("front3d"), name


@pytest.mark.parametrize("name", sorted(INFERENCE))
def test_inference_starts_from_the_relation_stage(configs: dict[str, Any], name: str) -> None:
    config = configs[name]
    assert "relation_scene_gcn_front3d" in config["weight"][0]
    assert "horizonnet" in config["weight"][1]
    assert config["model"]["detector"]["weight"].startswith("/PATH/TO/")


@pytest.mark.parametrize(
    ("name", "shape_weight"),
    [
        ("pano3d_front3d_im3d.yaml", "ldif_front3d"),
        ("shape500_im3d.yaml", "ldif_front3d"),
        ("pano3d_front3d_total3d.yaml", "mgnet_front3d"),
        ("shape500_total3d.yaml", "mgnet_front3d"),
    ],
)
def test_without_relations_the_shape_head_is_given_explicitly(
    configs: dict[str, Any], name: str, shape_weight: str
) -> None:
    config = configs[name]
    assert "scene_gcn" not in config["model"]
    assert shape_weight in config["weight"][2]


@pytest.mark.parametrize(("name", "steps"), [("pano3d_front3d.yaml", 100), ("pano3d_front3d_s20.yaml", 20)])
def test_relation_optimization_steps(configs: dict[str, Any], name: str, steps: int) -> None:
    assert configs[name]["model"]["scene_gcn"]["optimize_steps"] == steps


def test_inference_output_folders_are_distinct(configs: dict[str, Any]) -> None:
    paths = [configs[name]["log"]["path"] for name in INFERENCE]
    assert len(set(paths)) == len(paths)
