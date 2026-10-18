from pathlib import Path

from benchmark_toolbox.config import load_experiment_config
from benchmark_toolbox.datasets import BaseDatasetLoader
from benchmark_toolbox.evaluator import Evaluator
from benchmark_toolbox.metrics import BaseMetric
from benchmark_toolbox.models import BaseSceneEstimator


def test_example_pipeline(repo_root: Path, tmp_path: Path) -> None:
    config = load_experiment_config(repo_root / "configs/demo/smoke.json")
    config = config.__class__(
        experiment_name=config.experiment_name,
        seed=config.seed,
        output_dir=tmp_path,
        model=config.model,
        dataset=config.dataset,
        metrics=config.metrics,
        source_path=config.source_path,
    )
    estimator = BaseSceneEstimator.create(
        config.model.type, dict(config.model.parameters)
    )
    dataset = BaseDatasetLoader.create(
        config.dataset.type, dict(config.dataset.parameters)
    )
    metrics = [BaseMetric.create(m.type, dict(m.parameters)) for m in config.metrics]

    result = Evaluator(config, estimator, dataset, metrics).run()

    assert result.scene_count == 1
    assert result.summary["layout_iou_3d"]["mean"] == 1.0
    assert result.summary["object_map"]["mean"] == 1.0
    assert result.summary["collision_rate"]["mean"] == 0.0
    assert (tmp_path / "report.md").exists()
