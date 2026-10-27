from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pano3dbench.environments import ENVIRONMENT_MANAGERS, load_environment_spec
from pano3dbench.provenance import git_revision

LOGGER = logging.getLogger("pano3dbench.predictions")


@dataclass(frozen=True)
class RunnerLaunch:
    command: tuple[str, ...]
    batch: bool
    environment_name: str
    provenance: dict[str, Any]
    cwd: Path | None = None
    env: dict[str, str] | None = None


def usable_wrapper(wrapper: Sequence[str]) -> tuple[str, ...]:
    wrapper = tuple(str(part) for part in wrapper)
    if not wrapper:
        return ()
    if shutil.which(wrapper[0]) is not None:
        return wrapper
    LOGGER.warning(
        "Launch wrapper '%s' is not on PATH — skipping (headless mode). "
        "If the method needs a display, install it or run on a node with X/EGL.",
        wrapper[0],
    )
    return ()


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "runners").is_dir():
            return parent
    return here.parents[2]


def _resolve_entry(entry: str) -> str:
    path = Path(entry).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_project_root() / path).resolve())


def launch_from_environment(
    environment: Any, runner_args: Sequence[Any] = ()
) -> RunnerLaunch:
    spec = load_environment_spec(environment)
    manager = ENVIRONMENT_MANAGERS.create(spec.backend)
    if not manager.is_prepared(spec):
        raise RuntimeError(
            f"Environment '{spec.name}' ({spec.backend}) is not prepared. "
            f"Run: pano3dbench env prepare --env <spec>"
        )
    handle = manager.resolve(spec)
    requested_wrapper = tuple(handle.wrapper)
    wrapper = usable_wrapper(requested_wrapper)
    extra_args = tuple(str(argument) for argument in runner_args)
    command = (
        wrapper
        + tuple(handle.command_prefix)
        + (_resolve_entry(spec.runner.entry),)
        + tuple(spec.runner.args)
        + extra_args
        + ("--request", "{request}", "--output", "{output}")
    )

    fingerprint_root = manager.repo_dir(spec) if spec.repo else None
    provenance: dict[str, Any] = {
        "environment": spec.name,
        "backend": spec.backend,
        "variant": spec.variant,
        "fingerprint": spec.fingerprint(fingerprint_root),
        "runner": spec.runner.entry,
    }
    if extra_args:
        provenance["runner_args"] = list(extra_args)
    if wrapper:
        provenance["wrapper"] = list(wrapper)
    elif requested_wrapper:
        provenance["wrapper_skipped"] = list(requested_wrapper)
    if spec.repo:
        commit = git_revision(manager.repo_dir(spec))
        if commit:
            provenance["repo_commit"] = commit
    if spec.checkpoints:
        provenance["checkpoints"] = [
            {"dest": item.dest, "sha256": item.sha256} for item in spec.checkpoints
        ]

    return RunnerLaunch(
        command=command,
        batch=spec.runner.batch,
        environment_name=spec.name,
        provenance=provenance,
        cwd=handle.cwd,
        env=dict(handle.env) if handle.env else None,
    )


def launch_from_command(
    command: Any, batch: bool, environment_name: str
) -> RunnerLaunch:
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise ValueError(
            "Provide either 'environment' (spec) or 'command' "
            "(array of process arguments)"
        )
    arguments = tuple(str(argument) for argument in command)
    return RunnerLaunch(
        command=arguments,
        batch=bool(batch),
        environment_name=str(environment_name),
        provenance={"environment": str(environment_name), "command": list(arguments)},
    )


def run_runner(
    launch: RunnerLaunch,
    request: dict[str, Any],
    work_dir: Path,
    timeout_seconds: float,
) -> Any:
    request_path = work_dir / "request.json"
    output_path = work_dir / "output.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    command = [
        argument.replace("{request}", str(request_path)).replace(
            "{output}", str(output_path)
        )
        for argument in launch.command
    ]
    run_env = None
    if launch.env is not None:
        run_env = {**os.environ, **launch.env}
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=str(launch.cwd) if launch.cwd is not None else None,
        env=run_env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Model runner failed with code {completed.returncode}.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not output_path.exists():
        raise RuntimeError(
            "Model runner succeeded but did not create the requested output file"
        )
    with output_path.open(encoding="utf-8") as output_file:
        return json.load(output_file)
