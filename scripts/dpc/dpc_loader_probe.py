#!/usr/bin/env python3
import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())
    sys.argv = ["main.py", args.config, "--mode", args.mode]
    import main as entry
    from configs.config_utils import CONFIG
    from utils.project_utils import load_dataloader

    cfg = CONFIG(entry.parse_args())
    loader = load_dataloader(cfg.config, mode=args.mode)
    print("batches", len(loader), flush=True)
    for step, batch in enumerate(loader):
        est, gt = batch[0], batch[1]
        est_scenes, gt_scenes = (batch[2], batch[3]) if len(batch) > 3 else ([], [])
        lost = [name for name, d in (("est", est), ("gt", gt)) if "objs" not in d or not d["objs"]]
        if lost:
            print("step", step, "no objs in", lost, "est keys", sorted(est), "gt keys", sorted(gt), flush=True)
            print("scenes", [s.data.get("name") for s in gt_scenes], "batch len", len(batch), flush=True)
            print("est objs per scene", [len(s.data.get("objs", [])) for s in est_scenes],
                  "gt objs per scene", [len(s.data.get("objs", [])) for s in gt_scenes], flush=True)
            return 1
        if step % 100 == 0:
            print("step", step, "ok", flush=True)
        if step >= args.steps:
            print("probe passed", step, flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
