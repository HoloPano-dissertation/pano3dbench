import json
from pathlib import Path

import pytest
import yaml

from pano3dbench.config import load_experiment_config
from pano3dbench.datasets import BaseDatasetLoader, ManifestDatasetLoader
from pano3dbench.models import BaseSceneEstimator, SubprocessSceneEstimator
from pano3dbench.registry import ComponentRegistry

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
EXAMPLE_CONFIGS = sorted(
    path
    for folder in (CONFIGS_DIR / "demo", CONFIGS_DIR / "igibson" / "runs", CONFIGS_DIR / "front3d" / "runs")
    if folder.is_dir()
    for path in folder.iterdir()
    if path.is_file()
)


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_yaml(path: Path, payload: dict) -> Path:
    with path.open("w", encoding="utf-8") as o:
        yaml.dump(payload, o)
    return path


@pytest.mark.parametrize("name", ["manifest", "igibson"])
def test_dataset_aliases_resolve_to_one_implementation(name: str) -> None:
    loader = BaseDatasetLoader.create(name, {"manifest": str(Path(__file__))})
    assert isinstance(loader, ManifestDatasetLoader)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("name", ["subprocess", "dpc"])
def test_model_aliases_resolve_to_one_implementation(name: str) -> None:
    estimator = BaseSceneEstimator.create(name, {"command": ["true"]})
    assert isinstance(estimator, SubprocessSceneEstimator)  # ty: ignore[invalid-argument-type]


def test_alias_requires_a_registered_target() -> None:
    registry: ComponentRegistry[object] = ComponentRegistry("widget")
    with pytest.raises(ValueError, match="Cannot alias unknown"):
        registry.alias("missing", "other")


def test_alias_does_not_overwrite_an_existing_name() -> None:
    registry: ComponentRegistry[object] = ComponentRegistry("widget")
    registry.register("first")(lambda _parameters: "first")
    registry.register("second")(lambda _parameters: "second")
    with pytest.raises(ValueError, match="is already registered"):
        registry.alias("first", "second")


def test_relative_paths_resolve_against_the_config_directory(tmp_path: Path) -> None:
    (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")
    config_path = write_json(
        tmp_path / "experiment.json",
        {
            "experiment_name": "t",
            "model": {"type": "fixture", "parameters": {"prediction_dir": "pred"}},
            "dataset": {
                "type": "manifest",
                "parameters": {"manifest": "manifest.jsonl"},
            },
            "metrics": [
                {"type": "mesh_chamfer", "parameters": {"mesh_root": "meshes"}}
            ],
        },
    )
    config = load_experiment_config(config_path)

    assert config.metrics[0].parameters["mesh_root"] == str(tmp_path / "meshes")
    assert config.model.parameters["prediction_dir"] == str(tmp_path / "pred")


def test_output_dir_resolution(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "deep"
    config_dir.mkdir(parents=True)
    config_path = write_yaml(
        config_dir / "conf.yaml",
        {
            "output_dir": "../../artifacts/dpc",
            "model": {
                "type": "dpc",
                "parameters": {
                    "environment": "../environments/dpc.yaml",
                },
            },
            "dataset": {
                "type": "igibson",
                "parameters": {
                    "manifest": "../../data/examples/manifest.jsonl",
                },
            },
            "metrics": [
                {"type": "layout_iou_3d"},
                {"type": "object_map", "parameters": {"iou_threshold": 0.5}},
                {"type": "collision_rate"},
            ],
        },
    )

    config = load_experiment_config(config_path)
    assert config.output_dir == tmp_path / "artifacts" / "dpc"


def test_per_class_expands_into_one_row_per_class(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "c.json",
        {
            "model": {"type": "fixture", "parameters": {}},
            "dataset": {"type": "manifest", "parameters": {}},
            "metrics": [
                {
                    "type": "object_map_dataset",
                    "parameters": {
                        "iou_threshold": 0.15,
                        "classes": ["chair", "sofa"],
                        "per_class": True,
                        "name": "ap",
                    },
                }
            ],
        },
    )
    config = load_experiment_config(path)

    assert [
        (m.parameters["name"], m.parameters["classes"]) for m in config.metrics
    ] == [
        ("ap_chair", ["chair"]),
        ("ap_sofa", ["sofa"]),
    ]
    assert all(m.parameters["iou_threshold"] == 0.15 for m in config.metrics)


def test_metrics_are_inherited_and_locally_overridable(tmp_path: Path) -> None:
    write_json(
        tmp_path / "protocol.json",
        {
            "seed": 7,
            "metrics": [{"type": "collision_rate"}],
            "model": {"type": "fixture", "parameters": {"prediction_dir": "p"}},
            "dataset": {"type": "manifest", "parameters": {"manifest": "m"}},
        },
    )
    method_path = write_json(
        tmp_path / "method.json",
        {
            "extends": "protocol.json",
            "experiment_name": "method",
            "dataset": {"type": "manifest", "parameters": {"manifest": "other.jsonl"}},
        },
    )

    config = load_experiment_config(method_path)

    assert config.seed == 7
    assert config.experiment_name == "method"
    assert [m.type for m in config.metrics] == ["collision_rate"]
    assert config.dataset.parameters["manifest"] == str(tmp_path / "other.jsonl")
    assert config.model.parameters["prediction_dir"] == str(tmp_path / "p")


def test_cyclic_extends_is_reported(tmp_path: Path) -> None:
    a = write_json(tmp_path / "a.json", {"extends": "b.json"})
    write_json(tmp_path / "b.json", {"extends": "a.json"})
    with pytest.raises(ValueError, match="Cyclic"):
        load_experiment_config(a)


def test_example_configs_exist() -> None:
    assert EXAMPLE_CONFIGS, f"no example configs in {CONFIGS_DIR}"


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_every_example_config_loads(path: Path) -> None:
    config = load_experiment_config(path)
    assert config.metrics
