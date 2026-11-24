#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=Path)
    parser.add_argument("stage1", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.stage1
    out.mkdir(parents=True, exist_ok=True)

    member = {}
    for split in ("train", "val", "test"):
        for room in (args.exp / "splits" / f"{split}.txt").read_text().split():
            member[room] = split
    excluded_rows = args.exp / "splits" / "excluded_rooms.jsonl"
    if excluded_rows.is_file():
        for line in excluded_rows.read_text().splitlines():
            if line.strip():
                member.setdefault(json.loads(line)["room_id"], "excluded")
    found = {"train": [], "val": [], "test": [], "excluded": [], "unknown": []}
    empty = dict.fromkeys(found, 0)
    for path in args.stage1.glob("*/*/data.pkl"):
        folder = path.parent
        house, stem = folder.parts[-2], folder.parts[-1]
        split = member.get(f"{house}/{stem.rsplit('__', 1)[0]}", "unknown")
        with open(folder / "gt.pkl", "rb") as handle:
            if not pickle.load(handle).get("objs"):
                empty[split] += 1
                continue
        found[split].append(str(path.relative_to(args.stage1)))
    for name, ours in (("train", "train"), ("test", "val")):
        rows = sorted(found[ours])
        (out / f"{name}.json").write_text(json.dumps(rows, indent=1))
        print(f"{name}.json <- {ours}: {len(rows)} cameras")
    print("skipped without objects:", empty)
    print("unknown:", len(found["unknown"]), "| from test:", len(found["test"]),
          "| from excluded rooms:", len(found["excluded"]))
    return 1 if found["unknown"] or found["test"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
