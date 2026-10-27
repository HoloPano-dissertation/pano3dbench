from pathlib import Path

import pytest

from pano3dbench.geometry.boxes import BoundingBox3D
from pano3dbench.scene import SceneSample
from pano3dbench.predictions import RunnerPredictionSource
from pano3dbench.predictions.runner_process import usable_wrapper


@pytest.mark.parametrize(
    ("wrapper", "expected"),
    [
        pytest.param(["env", "-i"], ("env", "-i"), id="executable-present"),
        pytest.param(
            ["pano3dbench-no-such-binary-xyz", "-a"], (), id="executable-missing"
        ),
        pytest.param([], (), id="empty"),
    ],
)
def testusable_wrapper(wrapper: list[str], expected: tuple[str, ...]) -> None:
    assert usable_wrapper(wrapper) == expected


def test_json_protocol(fake_command: list[str]) -> None:
    estimator = RunnerPredictionSource(
        command=fake_command, environment_name="test-environment"
    )
    sample = SceneSample(sample_id="scene-001", input_path=Path(__file__))

    output = estimator.predict(sample)

    assert output.layout == BoundingBox3D((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert output.metadata["sample_id"] == "scene-001"
    assert output.metadata["environment"] == "test-environment"
