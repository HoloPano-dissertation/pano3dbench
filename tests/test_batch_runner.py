import argparse
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NoReturn

import _common as rc
import pytest

from benchmark_toolbox import SceneSample
from benchmark_toolbox.config import ComponentConfig, ExperimentConfig
from benchmark_toolbox.datasets.base import BaseDatasetLoader
from benchmark_toolbox.domain import SceneOutput
from benchmark_toolbox.environments import EnvironmentSpec
from benchmark_toolbox.evaluator import Evaluator
from benchmark_toolbox.metrics.geometry import CollisionRate
from benchmark_toolbox.models.base import BaseSceneEstimator
from benchmark_toolbox.models.subprocess import SubprocessSceneEstimator


# EXPLANATION: Wrong type hints of decorator confuse type checker
# TODO: Remove suppresions after fix
@pytest.fixture
def make_estimator(
    fake_command: list[str],
) -> Callable[[bool], SubprocessSceneEstimator]:  # ty: ignore[invalid-type-form]
    def _make(batch: bool) -> SubprocessSceneEstimator:  # ty: ignore[invalid-type-form]
        return SubprocessSceneEstimator(
            {
                "command": fake_command,
                "environment_name": "test-environment",
                "batch": batch,
            }
        )

    return _make


# --- request shape ------------------------------------------------------------


def test_read_requests_accepts_a_single_sample(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"sample_id": "a"}), encoding="utf-8")
    assert rc.read_requests(str(path)) == [{"sample_id": "a"}]


def test_read_requests_accepts_a_batch(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps({"samples": [{"sample_id": "a"}, {"sample_id": "b"}]}),
        encoding="utf-8",
    )
    assert [item["sample_id"] for item in rc.read_requests(str(path))] == ["a", "b"]


# --- batch estimator ----------------------------------------------------------


def test_batch_uses_a_single_launch_for_every_sample(
    make_estimator: Callable[[bool], SubprocessSceneEstimator],  # ty: ignore[invalid-type-form]
    make_samples: Callable[[int], list[SceneSample]],
) -> None:
    samples = make_samples(4)
    outputs = list(make_estimator(True).predict_many(samples))

    assert [o.metadata["sample_id"] for o in outputs] == [s.sample_id for s in samples]
    launches = {o.metadata["launch"] for o in outputs}
    assert len(launches) == 1, "the batch was split across processes"


def test_without_batch_each_sample_costs_its_own_launch(
    make_estimator: Callable[[bool], SubprocessSceneEstimator],  # ty: ignore[invalid-type-form]
    make_samples: Callable[[int], list[SceneSample]],
) -> None:
    outputs = list(make_estimator(False).predict_many(make_samples(4)))
    assert len({o.metadata["launch"] for o in outputs}) == 4


def test_results_are_identical_either_way(
    make_estimator: Callable[[bool], SubprocessSceneEstimator],  # ty: ignore[invalid-type-form]
    make_samples: Callable[[int], list[SceneSample]],
) -> None:
    samples = make_samples(3)
    batched = list(make_estimator(True).predict_many(samples))
    one_by_one = list(make_estimator(False).predict_many(samples))

    assert len(batched) == len(one_by_one) == len(samples)
    for left, right in zip(batched, one_by_one):
        assert left.layout == right.layout
        assert left.objects == right.objects
        assert left.metadata["sample_id"] == right.metadata["sample_id"]


def test_answers_are_matched_by_sample_id_not_by_position(
    make_estimator: Callable[[bool], SubprocessSceneEstimator],  # ty: ignore[invalid-type-form]
    make_samples: Callable[[int], list[SceneSample]],
) -> None:
    reordered = list(reversed(make_samples(3)))
    outputs = list(make_estimator(True).predict_many(reordered))
    assert [o.metadata["sample_id"] for o in outputs] == [
        s.sample_id for s in reordered
    ]


PARTIAL_RUNNER = """
import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
payload = json.loads(Path(args.request).read_text())
# answers only the first sample
first = payload["samples"][0]
Path(args.output).write_text(json.dumps({"outputs": [
    {"layout": None, "objects": [], "relations": [],
     "metadata": {"sample_id": first["sample_id"]}}
]}))
"""


def test_a_dropped_sample_is_an_error_not_a_silent_gap(
    tmp_path: Path, make_samples: Callable[[int], list[SceneSample]]
) -> None:
    import sys

    runner = tmp_path / "partial_runner.py"
    runner.write_text(PARTIAL_RUNNER, encoding="utf-8")
    estimator = SubprocessSceneEstimator(
        {
            "command": [
                sys.executable,
                str(runner),
                "--request={request}",
                "--output={output}",
            ],
            "batch": True,
        }
    )
    with pytest.raises(RuntimeError, match="no prediction for 2"):
        list(estimator.predict_many(make_samples(3)))


# --- runner args --------------------------------------------------------------


def test_runner_args_are_appended_after_the_specs_own(
    load_env: Callable[[str], EnvironmentSpec],
) -> None:
    estimator = SubprocessSceneEstimator(
        {
            "environment": load_env("echo"),
            "runner_args": ["--pred-root", "out/pano3d_im3d"],
        }
    )
    command = estimator.command  # ty: ignore[unresolved-attribute]

    assert "--pred-root" in command
    assert command[command.index("--pred-root") + 1] == "out/pano3d_im3d"
    assert command.index("--pred-root") < command.index("--request")


def test_runner_args_are_recorded_in_provenance(
    load_env: Callable[[str], EnvironmentSpec],
) -> None:
    estimator = SubprocessSceneEstimator(
        {"environment": load_env("echo"), "runner_args": ["--model", "im3d-pano"]}
    )
    assert estimator.provenance()["runner_args"] == ["--model", "im3d-pano"]


def test_a_repeated_option_is_won_by_the_experiment_config() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="base.yaml")
    args, _ = parser.parse_known_args(
        [
            "--config",
            "configs/pano3d_igibson.yaml",
            "--config",
            "configs/pano3d_igibson_im3d.yaml",
        ]
    )
    assert args.config == "configs/pano3d_igibson_im3d.yaml"


def test_shipped_method_configs_select_distinct_prediction_roots(
    repo_root: Path,
) -> None:
    roots: dict[str, str] = {}
    for path in (repo_root / "configs/examples").glob("*.yaml"):
        config = ExperimentConfig.fromfile(path)
        arguments = list(config.model.parameters.get("runner_args", ()))
        if "--pred-root" in arguments:
            roots[path.name] = arguments[arguments.index("--pred-root") + 1]

    assert roots, "no method config passes a prediction root"
    assert len(set(roots.values())) == len(roots), f"prediction roots collide: {roots}"


@pytest.mark.parametrize(
    ("env_name", "expected"), [("dpc", True), ("echo", True), ("holopano", False)]
)
def test_spec_declares_batch_support(
    load_env: Callable[[str], EnvironmentSpec], env_name: str, expected: bool
) -> None:
    assert load_env(env_name).runner.batch is expected


# --- evaluator ----------------------------------------------------------------


# EXPLANATION: Wrong type hints of decorator confuse type checker
class _ListDataset(BaseDatasetLoader):
    def __init__(self, samples: list[CollisionRate]) -> None:  # ty: ignore[invalid-type-form]
        self._samples = samples

    def samples(self) -> Iterator[CollisionRate]:  # ty: ignore[invalid-type-form]
        return iter(self._samples)


class _CountingEstimator(BaseSceneEstimator):
    """Records how the evaluator asks for predictions."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def predict(self, sample) -> NoReturn:  # noqa: ANN001, ARG002
        raise AssertionError("the evaluator must go through predict_many")

    def predict_many(self, samples) -> list[SceneOutput]:  # noqa: ANN001
        self.batches.append(len(samples))
        return [SceneOutput(layout=None) for _ in samples]


class _StreamWatchingEstimator(BaseSceneEstimator):
    def __init__(self, predictions_dir: Path) -> None:
        self.predictions_dir = predictions_dir
        self.seen: list[int] = []

    def predict(self, sample) -> SceneOutput:  # noqa: ANN001, ARG002
        existing = (
            len(list(self.predictions_dir.glob("*.json")))
            if self.predictions_dir.exists()
            else 0
        )
        self.seen.append(existing)
        return SceneOutput(layout=None)


def _config(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="t",
        seed=42,
        output_dir=output_dir,
        model=ComponentConfig(type="x", parameters={}),
        dataset=ComponentConfig(type="x", parameters={}),
        metrics=(),
        source_path=output_dir / "c.json",
    )


def test_predictions_are_written_as_they_are_produced(
    tmp_path: Path, make_samples: Callable[[int], list[SceneSample]]
) -> None:
    estimator = _StreamWatchingEstimator(tmp_path / "predictions")
    Evaluator(
        _config(tmp_path),
        estimator,
        _ListDataset(make_samples(4)),
        [CollisionRate()],  # ty: ignore[missing-argument]
    ).run()

    assert estimator.seen == [0, 1, 2, 3]


def test_evaluator_offers_the_whole_set_to_the_estimator(
    tmp_path: Path, make_samples: Callable[[int], list[SceneSample]]
) -> None:
    estimator = _CountingEstimator()
    result = Evaluator(
        _config(tmp_path),
        estimator,
        _ListDataset(make_samples(5)),
        [CollisionRate()],  # ty: ignore[missing-argument]
    ).run()

    assert estimator.batches == [5]
    assert result.scene_count == 5
