from collections.abc import Callable

import holopano_runner

from benchmark_toolbox.environments import EnvironmentSpec
from benchmark_toolbox.models import BaseSceneEstimator


def test_estimator_registered() -> None:
    assert "holopano" in BaseSceneEstimator.registry.names


def test_env_spec_loads(load_env: Callable[[str], EnvironmentSpec]) -> None:
    spec = load_env("holopano")
    assert spec.name == "HoloPano"
    assert spec.backend == "conda"
    assert spec.runner.entry == "runners/holopano_runner.py"


def test_converter_handles_dpc_like_output() -> None:
    native = {
        "objects": [
            {
                "label": "sofa",
                "score": 0.8,
                "bbox3d": {
                    "centroid": [0, 0, 0],
                    "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "coeffs": [1, 2, 3],
                },
            }
        ],
        "layout": [[0, 0, 0], [5, 3, 4]],
    }
    scene = holopano_runner.to_scene_output(native)

    assert len(scene["objects"]) == 1
    assert scene["objects"][0]["label"] == "sofa"
    bbox = scene["objects"][0]["bbox"]
    assert bbox["center"] == [0.0, 0.0, 0.0]
    assert bbox["size"] == [2.0, 4.0, 6.0]
    assert scene["layout"]["max_corner"] == [5.0, 3.0, 4.0]
