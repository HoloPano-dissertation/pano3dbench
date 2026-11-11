from pathlib import Path

import pytest

from pano3dbench.config import ExperimentConfig

RUNS = Path(__file__).resolve().parents[2] / "configs" / "front3d" / "runs"
CLASSES = {
    "table",
    "lighting",
    "chair",
    "storage unit",
    "bed",
    "sofa",
    "cabinet",
    "media unit",
    "plants",
    "other",
    "shelf",
    "accessory",
    "appliance",
    "electronics",
}


@pytest.mark.parametrize(
    "name",
    [
        "dpc_rescore.yaml",
        "dpc_s20_rescore.yaml",
        "im3d_rescore.yaml",
        "total3d_rescore.yaml",
        "gt_sanity.yaml",
    ],
)
def test_box_runs_report_ap_for_each_of_the_14_classes(name: str) -> None:
    config = ExperimentConfig.fromfile(RUNS / name)
    names = [metric.parameters.get("name", metric.type) for metric in config.metrics]

    assert {f"ap_{label}" for label in CLASSES} <= set(names)
    assert len(names) == len(set(names))
    assert "object_map_dataset" in names
    for metric in config.metrics:
        if metric.type.startswith("object_map"):
            assert metric.parameters["iou_threshold"] == 0.15
