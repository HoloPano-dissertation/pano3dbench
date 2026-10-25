import argparse
import json
import sys
from pathlib import Path


EXPORTS = ("splits/rooms.jsonl", "state/finalize.json", "state/ground_truth.json",
           "state/relations.json", "state/shape.json", "state/coco.json")


def require_training_approval(experiment_root, smoke=False):
    if smoke:
        return
    root = Path(experiment_root)
    path = root / "state" / "training_gate.json"
    if not path.is_file():
        raise RuntimeError("Full training blocked: no data-quality approval at %s" % path)
    report = json.loads(path.read_text())
    if report.get("training_approved") is not True:
        raise RuntimeError("Full training blocked by data-quality gate: %s" % report.get("reason", "unapproved"))
    approved = path.stat().st_mtime
    newer = [name for name in EXPORTS
             if (root / name).is_file() and (root / name).stat().st_mtime > approved]
    if newer:
        raise RuntimeError("Full training blocked: the set changed after its approval (%s); "
                           "review it again" % ", ".join(newer))


def main():
    parser = argparse.ArgumentParser(
        description="Check the data-quality gate before submitting a training job.")
    parser.add_argument("experiment_root", type=Path,
                        help="experiment directory holding state/training_gate.json")
    args = parser.parse_args()
    try:
        require_training_approval(args.experiment_root)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    print("Data-quality gate open: %s" % (args.experiment_root / "state/training_gate.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
