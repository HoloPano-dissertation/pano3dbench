import json

import pytest

pytest.importorskip("front3d_pano")
pytest.importorskip("PIL")

from validate_dpc import validate_dpc


@pytest.fixture
def exported_dataset(tmp_path):
    from PIL import Image
    CLASSES = ["bed", "chair", "table"]

    def write(path, data, jsonl=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r)+"\n" for r in data) if jsonl else json.dumps(data))

    rooms, mapping = [], []
    for split in ("train", "val", "test"):
        room_id = split+"-house/room"
        rooms.append({"room_id": room_id, "house_id": split+"-house", "split": split})
        records, images, annotations, dpc_paths = [], [], [], []
        for view in range(4):
            sample_id = room_id+"/"+str(view)
            rgb = tmp_path / "rgb" / split / (str(view)+".png")
            rgb.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1024, 512), "gray").save(rgb)
            gt = tmp_path / "gt" / split / (str(view)+".json")
            write(gt, {"layout": {"min_corner": [-1, -1, -1], "max_corner": [1, 1, 1]}, "objects": []})
            records.append({"sample_id": sample_id, "input": str(rgb), "ground_truth": str(gt),
                            "metadata": {"split": split}})
            images.append({"sample_id": sample_id, "id": view+1, "file_name": str(rgb), "width": 1024, "height": 512})
            annotations.append({"id": view+1, "image_id": view+1, "category_id": 1, "bbox": [0, 0, 1, 1],
                                "area": 1, "segmentation": {"size": [512, 1024], "counts": [0, 1, 512*1024-1]}})
            name = sample_id.replace("/", "__")
            link = tmp_path / "dpc_dataset/images" / split / (name+".png")
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(rgb)
            dpc_paths.append(link.relative_to(tmp_path / "dpc_dataset").as_posix())
            mapping.append({"dpc_name": name, "sample_id": sample_id, "split": split, "input": str(rgb), "ground_truth": str(gt)})
        for folder in ("manifests", "manifests_gt"):
            write(tmp_path / folder / (split+".jsonl"), records, True)
        write(tmp_path / "coco" / (split+".json"), {"images": images, "annotations": annotations,
              "categories": [{"id": i+1, "name": name, "supercategory": "furniture"} for i, name in enumerate(CLASSES)]})
        write(tmp_path / "dpc_dataset" / (split+".json"), dpc_paths)
        (tmp_path / "splits").mkdir(exist_ok=True)
        (tmp_path / "splits" / (split+".txt")).write_text(room_id+"\n")
    write(tmp_path / "splits/rooms.jsonl", rooms, True)
    write(tmp_path / "splits/excluded_rooms.jsonl", [], True)
    write(tmp_path / "dpc_dataset/sample_map.jsonl", mapping, True)
    return tmp_path


def test_dpc_links_match_the_panoramas_and_the_ground_truth(exported_dataset):
    assert validate_dpc(exported_dataset)["dpc_samples"] == {"train": 4, "val": 4, "test": 4}


def test_wrong_dpc_link_is_rejected(exported_dataset):
    link = next((exported_dataset / "dpc_dataset/images/train").glob("*.png"))
    link.unlink()
    link.symlink_to(exported_dataset / "rgb/test/0.png")
    with pytest.raises(ValueError, match="DPC source mismatch"):
        validate_dpc(exported_dataset)
