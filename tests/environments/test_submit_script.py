from collections.abc import Callable

from pano3dbench.environments import EnvironmentSpec, generate_submit_script


def test_slurm_script_for_gpu(load_env: Callable[[str], EnvironmentSpec]) -> None:
    script = generate_submit_script(
        "configs/environments/dpc.yaml",
        "configs/demo/dpc.yaml",
        spec=load_env("dpc"),
        scheduler="slurm",
        gpus=1,
    )
    assert "#!/bin/bash" in script
    assert "#SBATCH" in script
    assert "--gres=gpu:1" in script
    assert "module load anaconda" in script
    assert "pano3dbench env prepare --env configs/environments/dpc.yaml" in script
    assert "pano3dbench run --config configs/demo/dpc.yaml" in script


def test_bash_script_has_no_sbatch(load_env: Callable[[str], EnvironmentSpec]) -> None:
    script = generate_submit_script(
        "configs/environments/echo.yaml",
        spec=load_env("echo"),
        scheduler="bash",
        module_load="",
    )
    assert "#SBATCH" not in script
    assert "pano3dbench env prepare" in script
