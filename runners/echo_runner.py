#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner_protocol as protocol  # noqa: E402
import scene_payload as payload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="echo runner")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args, _unknown = parser.parse_known_args()

    requests = protocol.read_requests(args.request)
    protocol.log(f"[echo_runner] {len(requests)} sample(s)")

    def scene(request: dict) -> dict:
        return {
            "layout": payload.box([0.0, 0.0, 0.0], [4.0, 3.0, 5.0]),
            "objects": [
                payload.scene_object(
                    "obj-1", "chair", payload.box([1.0, 0.0, 1.0], [2.0, 1.0, 2.0]), 0.9
                )
            ],
            "relations": [],
            "metadata": {"model": "echo", "sample_id": request.get("sample_id")},
        }

    protocol.write_scenes(args.output, [scene(request) for request in requests])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
