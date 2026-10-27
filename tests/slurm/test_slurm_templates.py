import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SLURM = REPO / "scripts" / "slurm"
CONFIGS = REPO / "configs"
TRAININGS = (
    "bdb3d_estimation_front3d",
    "ldif_front3d",
    "mgnet_front3d",
    "relation_scene_gcn_front3d",
)
FAKE_SBATCH = """#!/bin/bash
count_file="$FAKE_SBATCH_DIR/count"
count=$(( $(cat "$count_file" 2>/dev/null || echo 100) + 1 ))
echo "$count" > "$count_file"
printf '%s\\0' "$count" "$@" >> "$FAKE_SBATCH_DIR/calls"
printf '\\n' >> "$FAKE_SBATCH_DIR/calls"
echo "$count"
"""


def _load_fill():
    spec = importlib.util.spec_from_file_location("fill_config", SLURM / "fill_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fill_config = _load_fill()


@pytest.fixture
def dpc(tmp_path: Path) -> Path:
    root = tmp_path / "dpc"
    for training in TRAININGS:
        for run in ("26100100000000", "26102000000000"):
            run_dir = root / "out" / training / run
            run_dir.mkdir(parents=True)
            (run_dir / "model_best.pth").write_text("", encoding="utf-8")
    (root / "configs").mkdir()
    return root


def test_run_placeholder_takes_the_latest_run_with_the_file(tmp_path: Path, dpc: Path) -> None:
    template = tmp_path / "t.yaml"
    text = "weight: ['/PATH/TO/out/ldif_front3d/RUN/model_best.pth']\n"
    filled = fill_config.fill(text, template, None, dpc)
    assert str(dpc / "out" / "ldif_front3d" / "26102000000000" / "model_best.pth") in filled


def test_missing_run_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No run of 'ldif_front3d'"):
        fill_config.fill(
            "weight: '/PATH/TO/out/ldif_front3d/RUN/model_best.pth'\n",
            tmp_path / "t.yaml",
            None,
            tmp_path,
        )


def test_relative_extends_and_output_dir_become_absolute(tmp_path: Path) -> None:
    template = tmp_path / "configs" / "runs" / "a.yaml"
    filled = fill_config.fill(
        "extends: ../protocols/boxes.yaml\noutput_dir: ../../artifacts/a\nseed: 1\n",
        template,
        None,
        None,
    )
    assert f"extends: {(tmp_path / 'configs' / 'protocols' / 'boxes.yaml').resolve()}" in filled
    assert f"output_dir: {(tmp_path / 'artifacts' / 'a').resolve()}" in filled


def test_unfilled_placeholder_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="placeholders left unfilled"):
        fill_config.fill("manifest: /PATH/TO/front3d/x.jsonl\n", tmp_path / "t.yaml", None, None)


@pytest.mark.parametrize(
    "config",
    sorted(
        [*(CONFIGS / "front3d" / "dpc").glob("*/*.yaml"), *(CONFIGS / "front3d" / "runs").glob("*.yaml"),
         *(CONFIGS / "igibson" / "runs").glob("*.yaml")]
    ),
    ids=lambda path: str(path.relative_to(CONFIGS)),
)
def test_every_shipped_config_fills_completely(tmp_path: Path, dpc: Path, config: Path) -> None:
    filled = fill_config.fill(config.read_text(encoding="utf-8"), config, tmp_path / "exp", dpc)
    assert "/PATH/TO" not in filled


@pytest.mark.parametrize("script", sorted(SLURM.glob("*.sh")) + sorted(SLURM.glob("*.sbatch")), ids=lambda path: path.name)
def test_scripts_parse(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", ["front3d.sh", "igibson.sh"])
def test_every_referenced_repository_file_exists(script: Path) -> None:
    text = (SLURM / script).read_text(encoding="utf-8")
    for relative in re.findall(r"\$TBX/([\w./-]+\.py)", text):
        assert (REPO / relative).is_file(), relative
    for relative in re.findall(r"\$SLURM_DIR/([\w.]+)", text):
        assert (SLURM / relative).is_file(), relative
    for relative in re.findall(r"\b((?:training|inference)/[\w]+\.yaml)", text):
        dataset = "front3d" if script == "front3d.sh" else "igibson"
        assert (CONFIGS / dataset / "dpc" / relative).is_file(), relative


def _submit(tmp_path: Path, dpc: Path, script: str, step: str, **env: str) -> list[list[str]]:
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "sbatch").write_text(FAKE_SBATCH, encoding="utf-8")
    (fake / "sbatch").chmod(0o755)
    record = tmp_path / "record"
    record.mkdir(exist_ok=True)
    completed = subprocess.run(
        ["bash", str(SLURM / script), step],
        env={
            **os.environ,
            "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}",
            "FAKE_SBATCH_DIR": str(record),
            "DPC": str(dpc),
            "DPC_PYTHON": sys.executable,
            "TBX_PYTHON": sys.executable,
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    calls = (record / "calls").read_text(encoding="utf-8").split("\n") if (record / "calls").exists() else []
    return [call.rstrip("\0").split("\0") for call in calls if call]


@pytest.fixture
def exp(tmp_path: Path) -> Path:
    root = tmp_path / "exp"
    (root / "state").mkdir(parents=True)
    (root / "state" / "classes.json").write_text(json.dumps({"classes": ["bed", "chair"]}), encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("step", "jobs"),
    [("components", 3), ("stage1", 2), ("heads", 3), ("gcn", 1), ("inference", 5), ("score", 8), ("shape", 8)],
)
def test_front3d_steps_submit_their_jobs(tmp_path: Path, dpc: Path, exp: Path, step: str, jobs: int) -> None:
    calls = _submit(tmp_path, dpc, "front3d.sh", step, EXP=str(exp), DETECTOR_INIT="/w.pkl")
    assert len(calls) == jobs
    for call in calls:
        assert any(argument.startswith("--output=") for argument in call)
    for call in calls[1:]:
        dependencies = [argument for argument in call if argument.startswith("--dependency=afterok:")]
        for dependency in dependencies:
            for job in dependency.split(":")[1:]:
                assert int(job) < int(call[0])
    assert all("/PATH/TO" not in path.read_text(encoding="utf-8") for path in (exp / "slurm_configs").glob("*.yaml"))


def test_front3d_chains_after_an_outside_job(tmp_path: Path, dpc: Path, exp: Path) -> None:
    calls = _submit(tmp_path, dpc, "front3d.sh", "gcn", EXP=str(exp), AFTER="42")
    assert "--dependency=afterok:42" in calls[0]


@pytest.mark.parametrize(("step", "jobs"), [("heads", 3), ("inference", 5), ("score", 11)])
def test_igibson_steps_submit_their_jobs(tmp_path: Path, dpc: Path, step: str, jobs: int) -> None:
    calls = _submit(tmp_path, dpc, "igibson.sh", step)
    assert len(calls) == jobs


def test_igibson_configs_are_copied_into_the_clone(tmp_path: Path, dpc: Path) -> None:
    _submit(tmp_path, dpc, "igibson.sh", "inference")
    assert (dpc / "configs" / "pano3d_igibson_total3d.yaml").is_file()
    assert (dpc / "configs" / "pano3d_igibson_im3d.yaml").is_file()


def test_unknown_step_prints_usage(tmp_path: Path, dpc: Path, exp: Path) -> None:
    completed = subprocess.run(
        ["bash", str(SLURM / "front3d.sh"), "nope"],
        env={**os.environ, "DPC": str(dpc), "DPC_PYTHON": sys.executable, "TBX_PYTHON": sys.executable, "EXP": str(exp)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "usage" in completed.stderr
