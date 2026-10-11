"""Component lookup by config name, and path resolution of config parameters."""

import json
from pathlib import Path

import pytest

from benchmark_toolbox.config import ConfigError, ExperimentConfig
from benchmark_toolbox.datasets import BaseDatasetLoader, ManifestDatasetLoader
from benchmark_toolbox.models import BaseSceneEstimator, SubprocessSceneEstimator
from benchmark_toolbox.registry import ComponentRegistry

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "configs" / "examples"
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIR.iterdir()) if EXAMPLES_DIR.is_dir() else []


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- registry -----------------------------------------------------------------


# EXPLANATION: Wrong type hints of decorator confuse type checker
# TODO: Remove suppresions after fix
@pytest.mark.parametrize("name", ["manifest", "igibson"])
def test_dataset_aliases_resolve_to_one_implementation(name: str) -> None:
    loader = BaseDatasetLoader.create(name, {"manifest": str(Path(__file__))})
    assert isinstance(loader, ManifestDatasetLoader)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("name", ["subprocess", "dpc", "holopano"])
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


# --- config paths -------------------------------------------------------------


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
    config = ExperimentConfig.fromfile(config_path)

    assert config.metrics[0].parameters["mesh_root"] == str(tmp_path / "meshes")
    assert config.model.parameters["prediction_dir"] == str(tmp_path / "pred")


# --- protocol inheritance -----------------------------------------------------


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
    config = ExperimentConfig.fromfile(path)

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

    config = ExperimentConfig.fromfile(method_path)

    assert config.seed == 7
    assert config.experiment_name == "method"
    assert [m.type for m in config.metrics] == ["collision_rate"]
    assert config.dataset.parameters["manifest"] == str(tmp_path / "other.jsonl")
    assert config.model.parameters["prediction_dir"] == str(tmp_path / "p")


def test_cyclic_extends_is_reported(tmp_path: Path) -> None:
    a = write_json(tmp_path / "a.json", {"extends": "b.json"})
    write_json(tmp_path / "b.json", {"extends": "a.json"})
    with pytest.raises(ConfigError, match="Cyclic"):
        ExperimentConfig.fromfile(a)


# --- shipped configs ----------------------------------------------------------


def test_example_configs_exist() -> None:
    assert EXAMPLE_CONFIGS, f"no example configs in {EXAMPLES_DIR}"


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_every_example_config_loads(path: Path) -> None:
    config = ExperimentConfig.fromfile(path)
    assert config.metrics
