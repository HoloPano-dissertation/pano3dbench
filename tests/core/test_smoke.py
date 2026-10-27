from pathlib import Path

from pano3dbench.config import ExperimentConfig
from pano3dbench.evaluation import Evaluator
from pano3dbench.experiment import (
    create_dataset,
    create_metrics,
    create_prediction_source,
)


def test_example_pipeline(repo_root: Path, tmp_path: Path) -> None:
    config = ExperimentConfig.fromfile(repo_root / "configs/demo/smoke.json")
    config = config.__class__(
        experiment_name=config.experiment_name,
        seed=config.seed,
        output_dir=tmp_path,
        model=config.model,
        dataset=config.dataset,
        metrics=config.metrics,
        source_path=config.source_path,
    )
    result = Evaluator(
        config,
        create_dataset(config.dataset),
        create_prediction_source(config.model, config.seed),
        create_metrics(config.metrics),
    ).run()

    assert result.scene_count == 1
    assert result.summary["layout_iou_3d"]["mean"] == 1.0
    assert result.summary["object_map"]["mean"] == 1.0
    assert result.summary["collision_rate"]["mean"] == 0.0
    assert (tmp_path / "report.md").exists()
