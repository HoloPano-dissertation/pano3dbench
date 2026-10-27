from __future__ import annotations

import logging
import os
import shlex
import shutil
import sys
from pathlib import Path

from pano3dbench.environments.base import ENVIRONMENT_MANAGERS, EnvHandle, EnvironmentManager
from pano3dbench.environments.patches import apply_patches
from pano3dbench.environments.spec import EnvironmentSpec
from pano3dbench.environments.steps import (
    clone_repo,
    download_checkpoint,
    run_step,
)

LOGGER = logging.getLogger("pano3dbench.environments")


@ENVIRONMENT_MANAGERS.register("venv")
class VenvEnvironmentManager(EnvironmentManager):
    def _venv_dir(self, spec: EnvironmentSpec) -> Path:
        return spec.workspace / "venvs" / spec.name

    def _venv_python(self, spec: EnvironmentSpec) -> Path:
        directory = self._venv_dir(spec)
        if os.name == "nt":
            return directory / "Scripts" / "python.exe"
        return directory / "bin" / "python"

    def environment_exists(self, spec: EnvironmentSpec) -> bool:
        return self._venv_python(spec).exists()

    def _base_python(self, spec: EnvironmentSpec) -> str:
        if spec.python:
            candidates = [
                f"python{spec.python}",
                f"python{spec.python.split('.')[0]}",
                spec.python,
            ]
            for candidate in candidates:
                found = shutil.which(candidate)
                if found:
                    return found
            LOGGER.warning(
                "Python %s not found, using the current interpreter %s",
                spec.python,
                sys.executable,
            )
        return sys.executable

    def prepare(self, spec: EnvironmentSpec) -> EnvHandle:
        venv_dir = self._venv_dir(spec)
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        if not self.environment_exists(spec):
            run_step([self._base_python(spec), "-m", "venv", venv_dir])

        python = str(self._venv_python(spec))
        if spec.requirements or spec.pip:
            run_step([python, "-m", "pip", "install", "--upgrade", "pip"])
        for requirement in spec.requirements:
            run_step([python, "-m", "pip", "install", "-r", spec.base_dir / requirement])
        for item in spec.pip:
            run_step([python, "-m", "pip", "install"] + shlex.split(item))

        if spec.repo:
            clone_repo(spec.repo, self.repo_dir(spec))
        if spec.patches:
            apply_patches(spec.patches, self.repo_dir(spec))
        for checkpoint in spec.checkpoints:
            download_checkpoint(checkpoint, self.repo_dir(spec))

        self._write_marker(spec, repo_dir=self.repo_dir(spec) if spec.repo else None)
        return self.resolve(spec)

    def resolve(self, spec: EnvironmentSpec) -> EnvHandle:
        prefix = (str(self._venv_python(spec)),)
        cwd = self.repo_dir(spec) if spec.repo else None
        env = self._runtime_env(spec)
        return EnvHandle(
            command_prefix=prefix,
            cwd=cwd,
            env=env,
            wrapper=tuple(spec.runner.wrapper),
        )
