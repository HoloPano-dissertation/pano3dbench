import json
from pathlib import Path

import pytest
import yaml

from pano3dbench.config import ConfigError, ExperimentConfig
from pano3dbench.datasets import DATASET_LOADERS, ManifestDatasetLoader
from pano3dbench.predictions import PREDICTION_SOURCES, RunnerPredictionSource
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
    loader = DATASET_LOADERS.create(name, {"manifest": str(Path(__file__))})
    assert isinstance(loader, ManifestDatasetLoader)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("name", ["subprocess", "dpc"])
def test_model_aliases_resolve_to_one_implementation(name: str) -> None:
    estimator = PREDICTION_SOURCES.create(name, {"command": ["true"]})
    assert isinstance(estimator, RunnerPredictionSource)  # ty: ignore[invalid-argument-type]


def test_unknown_component_parameter_is_reported() -> None:
    with pytest.raises(ValueError, match="Invalid parameters for dataset loader"):
        DATASET_LOADERS.create("manifest", {"manifest": "m.jsonl", "manifets": "x"})


def test_missing_component_parameter_is_reported() -> None:
    with pytest.raises(ValueError, match="missing a required argument: 'manifest'"):
        DATASET_LOADERS.create("manifest", {})


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
    config = ExperimentConfig.fromfile(config_path)

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

    config = ExperimentConfig.fromfile(config_path)
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


def _minimal(tmp_path: Path, **overrides: object) -> dict:
    payload: dict = {
        "model": {"type": "fixture", "parameters": {"prediction_dir": "pred"}},
        "dataset": {"type": "manifest", "parameters": {"manifest": "m.jsonl"}},
        "metrics": [{"type": "layout_iou_3d"}],
    }
    payload.update(overrides)
    return payload


def test_home_in_a_parameter_path_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    payload = _minimal(tmp_path)
    payload["dataset"]["parameters"]["manifest"] = "~/data/m.jsonl"
    config = ExperimentConfig.fromfile(write_json(tmp_path / "c.json", payload))
    assert config.dataset.parameters["manifest"] == str(tmp_path / "home" / "data" / "m.jsonl")


def test_home_in_extends_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    write_json(home / "base.json", _minimal(tmp_path))
    config = ExperimentConfig.fromfile(write_json(tmp_path / "c.json", {"extends": "~/base.json"}))
    assert config.metrics[0].type == "layout_iou_3d"


def test_absolute_paths_are_kept_as_written(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    payload = _minimal(tmp_path, output_dir=str(link / "out"))
    payload["model"]["parameters"]["prediction_dir"] = str(link / "pred")
    config = ExperimentConfig.fromfile(write_json(tmp_path / "c.json", payload))
    assert config.model.parameters["prediction_dir"] == str(link / "pred")
    assert config.output_dir == link / "out"


@pytest.mark.parametrize("value", [None, 3, ["a", "b"], {"x": 1}])
def test_non_string_path_values_are_left_alone(tmp_path: Path, value: object) -> None:
    payload = _minimal(tmp_path)
    payload["model"]["parameters"]["checkpoint"] = value
    config = ExperimentConfig.fromfile(write_json(tmp_path / "c.json", payload))
    assert config.model.parameters["checkpoint"] == value


def test_a_metric_without_parameters_does_not_stop_path_resolution(tmp_path: Path) -> None:
    payload = _minimal(
        tmp_path,
        metrics=[
            {"type": "layout_iou_3d"},
            {"type": "mesh_chamfer", "parameters": {"reference_dir": "refs"}},
        ],
    )
    config = ExperimentConfig.fromfile(write_json(tmp_path / "c.json", payload))
    assert config.metrics[1].parameters["reference_dir"] == str(tmp_path / "refs")


def test_per_class_false_keeps_a_single_metric(tmp_path: Path) -> None:
    payload = _minimal(
        tmp_path,
        metrics=[{"type": "object_map", "parameters": {"per_class": False, "classes": ["bed", "chair"]}}],
    )
    config = ExperimentConfig.fromfile(write_json(tmp_path / "c.json", payload))
    assert [m.parameters.get("name") for m in config.metrics] == [None]
    assert config.metrics[0].parameters["classes"] == ["bed", "chair"]


def test_cyclic_extends_lists_files_in_order(tmp_path: Path) -> None:
    a = write_json(tmp_path / "a.json", {"extends": "b.json"})
    write_json(tmp_path / "b.json", {"extends": "c.json"})
    write_json(tmp_path / "c.json", {"extends": "a.json"})
    with pytest.raises(ConfigError) as error:
        ExperimentConfig.fromfile(a)
    chain = str(error.value).split(": ", 1)[1].split(" -> ")
    assert [Path(item).name for item in chain] == ["a.json", "b.json", "c.json", "a.json"]


@pytest.mark.parametrize("text", ["- a\n- b\n", "just text\n"])
def test_a_config_that_is_not_a_mapping_is_a_config_error(tmp_path: Path, text: str) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        ExperimentConfig.fromfile(path)


def test_example_configs_exist() -> None:
    assert EXAMPLE_CONFIGS, f"no example configs in {CONFIGS_DIR}"


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_every_example_config_loads(path: Path) -> None:
    config = ExperimentConfig.fromfile(path)
    assert config.metrics
