from __future__ import annotations

from pathlib import Path

from pano3dbench.config import ComponentConfig, ExperimentConfig
from pano3dbench.datasets import DATASET_LOADERS, DatasetLoader
from pano3dbench.evaluation import EvaluationResult, Evaluator
from pano3dbench.metrics import METRICS, Metric
from pano3dbench.predictions import PREDICTION_SOURCES, PredictionSource


def run_experiment(config_path: str | Path) -> EvaluationResult:
    config = ExperimentConfig.fromfile(config_path)
    evaluator = Evaluator(
        config=config,
        dataset=create_dataset(config.dataset),
        prediction_source=create_prediction_source(config.model, config.seed),
        metrics=create_metrics(config.metrics),
    )
    return evaluator.run()


def create_dataset(component: ComponentConfig) -> DatasetLoader:
    return DATASET_LOADERS.create(component.type, component.parameters)


def create_prediction_source(component: ComponentConfig, seed: int) -> PredictionSource:
    parameters = dict(component.parameters)
    parameters.setdefault("seed", seed)
    return PREDICTION_SOURCES.create(component.type, parameters)


def create_metrics(components: tuple[ComponentConfig, ...]) -> list[Metric]:
    return [METRICS.create(metric.type, metric.parameters) for metric in components]
