from pathlib import Path

import pytest

from pano3dbench.domain import BoundingBox3D, SceneSample
from pano3dbench.models.subprocess import (
    SubprocessSceneEstimator,
    _usable_wrapper,
)


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
def test_usable_wrapper(wrapper: list[str], expected: tuple[str, ...]) -> None:
    assert _usable_wrapper(wrapper) == expected


def test_json_protocol(fake_command: list[str]) -> None:
    estimator = SubprocessSceneEstimator(
        {"command": fake_command, "environment_name": "test-environment"}
    )
    sample = SceneSample(sample_id="scene-001", input_path=Path(__file__))

    output = estimator.predict(sample)

    assert output.layout == BoundingBox3D((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert output.metadata["sample_id"] == "scene-001"
    assert output.metadata["environment"] == "test-environment"
