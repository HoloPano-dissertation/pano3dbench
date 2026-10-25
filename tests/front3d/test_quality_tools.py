import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("front3d_pano")

np = pytest.importorskip("numpy")
pytest.importorskip("shapely")

QA = Path(__file__).resolve().parents[2] / "scripts" / "front3d" / "qa"


def write_fixture(path, indexed=True, translation=(0, 0, 0), matrix=None, mode=4):
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype="<f4")
    binary = points.tobytes() + np.asarray([0, 1, 2], dtype="<u2").tobytes() + b"\0\0"
    node = {"mesh": 0, "translation": list(translation)}
    if matrix is not None:
        node = {"mesh": 0, "matrix": matrix.T.reshape(-1).tolist()}
    primitive = {"attributes": {"POSITION": 0}, "mode": mode}
    if indexed:
        primitive["indices"] = 1
    document = {"asset": {"version": "2.0"}, "scene": 0,
                "scenes": [{"nodes": [0]}], "nodes": [{"translation": [10, 20, 30], "children": [1]}, node],
                "meshes": [{"primitives": [primitive]}], "buffers": [{"byteLength": len(binary)}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
                                {"buffer": 0, "byteOffset": 36, "byteLength": 6}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                              {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}]}
    content = json.dumps(document).encode()
    content += b" " * (-len(content) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28+len(content)+len(binary))
                     + struct.pack("<II", len(content), 0x4E4F534A) + content
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)


def test_image_flags(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    import quality
    path = tmp_path / "0.hdf5"
    with h5py.File(path, "w") as f:
        f["colors"] = np.zeros((16, 32, 3), dtype=np.uint8)
        f["depth"] = np.full((16, 32), np.inf)
    metrics, flags, image = quality.image_checks(path, 2)
    assert "mostly_dark" in flags and "missing_depth" in flags
    assert metrics["depth_valid_fraction"] == 0
    assert image.size == (32, 16)


def test_missing_layout_metadata_is_not_a_successful_audit(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    import audit as dataset_audit
    root, scene, report = tmp_path / "experiment", tmp_path / "source", tmp_path / "report"
    room = scene / "house" / "bedroom"
    rendered = root / "outputs" / "train" / "house" / "bedroom"
    room.mkdir(parents=True)
    rendered.mkdir(parents=True)
    (report / "previews").mkdir(parents=True)
    (report / "rooms").mkdir()
    write_fixture(room / "wall.glb")
    write_fixture(room / "ceil.glb", translation=(0, 2, 0))
    cameras = [[10.1+i*0.1, -30.2, 21] for i in range(4)]
    (rendered / "render.json").write_text(json.dumps({"camera_locations": cameras}))
    for index in range(4):
        with h5py.File(rendered / f"{index}.hdf5", "w") as f:
            f["colors"] = np.full((16, 32, 3), 50+index, dtype=np.uint8)
            f["depth"] = np.ones((16, 32))
    result = dataset_audit.audit(({"room_id": "house/bedroom", "split": "train", "room_dir": str(room)},
                                  str(root / "outputs"), str(report)))
    assert "audit_error" in result["flags"]
    assert not result["geometry_pass"]


def test_all_flagged_review_includes_each_failed_geometry_room(tmp_path):
    from PIL import Image
    report, output = tmp_path / "audit", tmp_path / "review"
    (report / "previews").mkdir(parents=True)
    rooms = []
    for index in range(20):
        preview = "previews/room-%02d.jpg" % index
        Image.new("RGB", (16, 16), "gray").save(report / preview)
        rooms.append({"room_id": "house/room-%02d" % index, "split": "train", "preview": preview,
                      "geometry_pass": False, "flags": ["inside_furniture_bbox"]})
    (report / "rooms.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rooms))
    subprocess.run([sys.executable, str(QA / "review.py"), str(report), str(output), "--all-flagged"], check=True)
    selected = json.loads((output / "selection.json").read_text())
    assert {r["room_id"] for r in selected} == {r["room_id"] for r in rooms}
