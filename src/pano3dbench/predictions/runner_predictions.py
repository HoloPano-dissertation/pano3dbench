from __future__ import annotations

import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pano3dbench.predictions.base import PREDICTION_SOURCES, PredictionSource
from pano3dbench.predictions.runner_process import (
    launch_from_command,
    launch_from_environment,
    run_runner,
)
from pano3dbench.scene import SceneOutput, SceneSample


@PREDICTION_SOURCES.register("subprocess")
class RunnerPredictionSource(PredictionSource):
    def __init__(
        self,
        environment: Any = None,
        command: Sequence[str] | None = None,
        runner_args: Sequence[Any] = (),
        batch: bool = False,
        environment_name: str = "external",
        timeout_seconds: float = 3600,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        if environment is not None:
            self.launch = launch_from_environment(environment, runner_args)
        else:
            self.launch = launch_from_command(command, batch, environment_name)
        self.timeout_seconds = float(timeout_seconds)

    @property
    def batch(self) -> bool:
        return self.launch.batch

    @property
    def environment_name(self) -> str:
        return self.launch.environment_name

    def provenance(self) -> dict[str, Any]:
        return dict(self.launch.provenance)

    def predict(self, sample: SceneSample) -> SceneOutput:
        with self._work_dir() as work_dir:
            return self._scene(self._run(self._request(sample), work_dir))

    def predict_many(self, samples: Sequence[SceneSample]) -> Iterable[SceneOutput]:
        if not self.batch or len(samples) <= 1:
            yield from super().predict_many(samples)
            return
        with self._work_dir() as work_dir:
            request = {"samples": [self._request(sample) for sample in samples]}
            payload = self._run(request, work_dir)
        by_id = _outputs_by_sample_id(payload)
        missing = [s.sample_id for s in samples if s.sample_id not in by_id]
        if missing:
            shown = ", ".join(missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise RuntimeError(
                f"The batch runner returned no prediction for {len(missing)} of "
                f"{len(samples)} samples: {shown}{more}"
            )
        for sample in samples:
            yield self._scene(by_id[sample.sample_id])

    def _request(self, sample: SceneSample) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "input_path": str(sample.input_path.resolve()),
            "metadata": dict(sample.metadata),
            "seed": self.seed,
        }

    def _run(self, request: dict[str, Any], work_dir: Path) -> Any:
        return run_runner(self.launch, request, work_dir, self.timeout_seconds)

    def _scene(self, payload: Any) -> SceneOutput:
        output = SceneOutput.from_dict(payload)
        metadata = dict(output.metadata)
        metadata.setdefault("environment", self.environment_name)
        return SceneOutput(
            layout=output.layout,
            objects=output.objects,
            relations=output.relations,
            metadata=metadata,
        )

    @contextmanager
    def _work_dir(self) -> Iterator[Path]:
        with tempfile.TemporaryDirectory(prefix="pano3dbench-") as temporary:
            yield Path(temporary)


def _outputs_by_sample_id(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or "outputs" not in payload:
        raise RuntimeError(
            "A batch runner must answer with {'outputs': [...]}, one entry per "
            "requested sample (see runners/README.md)"
        )
    by_id: dict[str, Any] = {}
    for entry in payload["outputs"]:
        sample_id = (entry.get("metadata") or {}).get("sample_id") or entry.get(
            "sample_id"
        )
        if sample_id is None:
            raise RuntimeError(
                "A batch runner's output entry must carry its sample_id "
                "(in metadata.sample_id, or as a top-level 'sample_id')"
            )
        by_id[str(sample_id)] = entry
    return by_id


PREDICTION_SOURCES.alias("subprocess", "dpc")
