from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pano3dbench.scene import SceneOutput


class ResultWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.predictions_dir = output_dir / "predictions"

    def prepare(self) -> None:
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def write_prediction(self, sample_id: str, prediction: SceneOutput) -> None:
        path = self.predictions_dir / f"{sample_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_scene_metrics(self, records: list[dict[str, Any]]) -> None:
        (self.output_dir / "metrics.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def write_summary(self, summary: dict[str, dict[str, Any]]) -> None:
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_run_manifest(self, manifest: dict[str, Any]) -> None:
        (self.output_dir / "run.json").write_text(
            json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str
            ),
            encoding="utf-8",
        )

    def write_report(
        self, experiment_name: str, summary: dict[str, dict[str, Any]]
    ) -> None:
        lines = [
            f"# Experiment: {experiment_name}",
            "",
            "| Metric | Count | Mean | Median | 95% CI (bootstrap) |",
            "|---|---:|---:|---:|---:|",
        ]
        lines.extend(_report_row(name, values) for name, values in summary.items())
        (self.output_dir / "report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def _report_row(metric_name: str, values: dict[str, Any]) -> str:
    if values.get("dataset_level"):
        return (
            f"| {metric_name} (dataset) | {values['count']} | "
            f"{values['value']:.6f} | — | — |"
        )
    if not values["count"]:
        return f"| {metric_name} | 0 | n/a | n/a | n/a |"
    return (
        f"| {metric_name} | {values['count']} | "
        f"{values['mean']:.6f} | {values['median']:.6f} | "
        f"[{values['ci95_low']:.6f}, {values['ci95_high']:.6f}] |"
    )
