from pathlib import Path

from benchmark_toolbox.domain import SceneSample
from benchmark_toolbox.environments import load_environment_spec
from benchmark_toolbox.environments.venv import VenvEnvironmentManager
from benchmark_toolbox.models.subprocess import SubprocessSceneEstimator


def test_prepare_and_run_echo_environment(tmp_path: Path, repo_root: Path) -> None:
    spec = load_environment_spec(
        {
            "name": "echo",
            "backend": "venv",
            "variant": "cpu",
            "workspace": str(tmp_path),
            "runner": {"entry": str(repo_root / "runners/echo_runner.py")},
        },
        base_dir=repo_root,
    )
    manager = VenvEnvironmentManager()
    assert not manager.is_prepared(spec)

    manager.prepare(spec)
    assert manager.is_prepared(spec)

    estimator = SubprocessSceneEstimator({"environment": spec})
    provenance = estimator.provenance()
    assert provenance["environment"] == "echo"
    assert provenance["backend"] == "venv"
    assert "fingerprint" in provenance

    output = estimator.predict(
        SceneSample(sample_id="scene-x", input_path=Path(__file__))
    )

    assert output.layout is not None
    assert len(output.objects) == 1
    assert output.metadata["sample_id"] == "scene-x"
    assert output.metadata["environment"] == "echo"
