#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path


def dpc_name(sample_id):
    house, room, view = sample_id.split("/")
    return f"test/{house}/{room}/{room}__{view}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    out = args.out or args.exp / "eval" / "shape500"
    (out / "gt").mkdir(parents=True, exist_ok=True)

    manifest = [json.loads(line) for line in (args.exp / "eval" / "manifest_test.jsonl").read_text().splitlines()
                if line.strip()]
    split = json.loads((args.exp / "dpc_scenes" / "test_split.json").read_text())
    by_dpc = {dpc_name(m["sample_id"]): m for m in manifest}
    if set(by_dpc) != set(split):
        raise SystemExit("manifest_test.jsonl and test_split.json disagree")
    chosen = sorted(random.Random(args.seed).sample(sorted(split), args.count))
    (out / "test_split_shape500.json").write_text(json.dumps(chosen, indent=1))
    (out / "views.txt").write_text("".join(by_dpc[c]["sample_id"] + "\n" for c in chosen))

    n_obj = n_shape = 0
    missing = []
    with open(out / "manifest_gt.jsonl", "w") as gt_out, open(out / "manifest_test.jsonl", "w") as test_out:
        for c in chosen:
            m = by_dpc[c]
            test_out.write(json.dumps(m) + "\n")
            gt = json.loads(Path(m["ground_truth"]).read_text())
            for o in gt["objects"]:
                n_obj += 1
                shape = o.get("attributes", {}).get("shape")
                if not shape:
                    continue
                watertight = os.path.join(os.path.dirname(shape), "mesh_watertight.ply")
                if os.path.exists(watertight):
                    o["attributes"]["shape"] = watertight
                    n_shape += 1
                else:
                    o["attributes"].pop("shape")
                    missing.append(watertight)
            target = out / "gt" / f"{m['sample_id']}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(gt))
            gt_out.write(json.dumps(dict(m, ground_truth=str(target))) + "\n")
    print("views", len(chosen), "objects", n_obj, "with watertight", n_shape, "missing", len(missing), missing[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
