#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=Path)
    parser.add_argument("--scenes", type=Path)
    parser.add_argument("--split-out", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    args = parser.parse_args()
    scenes = args.scenes or args.exp / "dpc_scenes"
    split_out = args.split_out or scenes / "test_split.json"
    manifest_out = args.manifest_out or args.exp / "eval" / "manifest_test.jsonl"

    folders, kept, empty = [], [], 0
    for line in (args.exp / "manifests_gt" / "test.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        house, room, view = json.loads(line)["sample_id"].split("/")
        folder = Path("test") / house / room / f"{room}__{view}"
        pkl = scenes / folder / "data.pkl"
        if not pkl.is_file():
            empty += 1
            continue
        with pkl.open("rb") as handle:
            if not pickle.load(handle).get("objs"):
                empty += 1
                continue
        folders.append(str(folder))
        kept.append(line)
    split_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    split_out.write_text(json.dumps(folders, indent=1))
    manifest_out.write_text("\n".join(kept) + "\n")
    print(f"test views: {len(folders)} | without objects: {empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
