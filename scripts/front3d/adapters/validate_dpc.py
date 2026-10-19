#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from front3d_pano.dataset.validate_dataset import by_sample, read_rows, require


def validate_dpc(root):
    root = Path(root)
    rooms = read_rows(root / "splits/rooms.jsonl")
    mapping = read_rows(root / "dpc_dataset/sample_map.jsonl")
    require(len({r["dpc_name"] for r in mapping}) == len(mapping), "Duplicate DPC names")
    require(all(r["split"] in ("train", "val", "test") for r in mapping), "Invalid DPC split")
    counts = {}
    for split in ("train", "val", "test"):
        room_ids = {r["room_id"] for r in rooms if r["split"] == split}
        expected = {room+"/"+str(view) for room in room_ids for view in range(4)}
        rgb = by_sample(read_rows(root / "manifests" / (split+".jsonl")), expected, "RGB")
        gt = by_sample(read_rows(root / "manifests_gt" / (split+".jsonl")), expected, "GT")
        mapped = by_sample([r for r in mapping if r["split"] == split], expected, "DPC")
        paths = json.loads((root / "dpc_dataset" / (split+".json")).read_text())
        expected_paths = {"images/"+split+"/"+r["dpc_name"]+".png" for r in mapped.values()}
        require(len(paths) == len(expected_paths) and set(paths) == expected_paths, "DPC split mismatch")
        for sample_id, record in mapped.items():
            link = root / "dpc_dataset/images" / split / (record["dpc_name"]+".png")
            require(link.is_file() and link.resolve() == Path(rgb[sample_id]["input"]).resolve()
                    and Path(record["input"]).resolve() == link.resolve()
                    and Path(record["ground_truth"]).resolve() == Path(gt[sample_id]["ground_truth"]).resolve(),
                    "DPC source mismatch")
        counts[split] = len(mapped)
    return {"ready": True, "dpc_samples": counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_dpc(args.experiment_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
