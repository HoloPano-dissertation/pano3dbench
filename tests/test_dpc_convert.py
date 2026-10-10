import pickle
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import dpc_runner
import pytest

IDENTITY = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]


def _bdb3d(centroid: tuple[int, int, int], size: tuple[int, int, int]) -> dict:
    return {"centroid": centroid, "basis": IDENTITY, "size": size}


def test_convert_data_pkl() -> None:
    data = {
        "objs": [
            {"classname": "chair", "score": 0.9, "bdb3d": _bdb3d((0, 0, 0), (2, 2, 2))},
            {"classname": "no-3d-box"},
        ],
        "layout": {
            "manhattan_world": [
                [2, 3, 4],
                [2, 3, -4],
                [2, -3, 4],
                [2, -3, -4],
                [-2, 3, 4],
                [-2, 3, -4],
                [-2, -3, 4],
                [-2, -3, -4],
            ],
        },
    }
    scene = dpc_runner.convert_data_pkl(data)

    assert len(scene["objects"]) == 1
    assert scene["objects"][0]["label"] == "chair"
    assert scene["objects"][0]["score"] == 0.9
    bbox = scene["objects"][0]["bbox"]
    assert bbox["center"] == [0.0, 0.0, 0.0]
    assert bbox["size"] == [2.0, 2.0, 2.0]
    assert bbox["basis"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert scene["layout"]["min_corner"] == [-2.0, -3.0, -4.0]
    assert scene["layout"]["max_corner"] == [2.0, 3.0, 4.0]


def test_convert_attaches_mesh_path_as_shape() -> None:
    data = {
        "objs": [
            {
                "classname": "chair",
                "mesh_path": "/abs/pred/s1/obj-0.ply",
                "bdb3d": _bdb3d((0, 0, 0), (2, 2, 2)),
            },
            {
                "classname": "table",
                "mesh": object(),
                "bdb3d": _bdb3d((5, 5, 5), (1, 1, 1)),
            },
        ],
        "layout": None,
    }
    scene = dpc_runner.convert_data_pkl(data)

    assert scene["objects"][0]["attributes"]["shape"] == "/abs/pred/s1/obj-0.ply"
    assert "shape" not in scene["objects"][1]["attributes"]


def test_find_data_pkl_matches_sample_id(tmp_path: Path) -> None:
    for scene in ("scene-A", "scene-B"):
        pkl = tmp_path / "run1" / "visualization" / scene / "data.pkl"
        pkl.parent.mkdir(parents=True, exist_ok=True)
        pkl.write_bytes(b"")

    pkl_data_b = dpc_runner.find_data_pkl("scene-B", search_root=str(tmp_path))
    pkl_data_a = dpc_runner.find_data_pkl("scene-A", search_root=str(tmp_path))
    pkl_data_missing = dpc_runner.find_data_pkl("missing", search_root=str(tmp_path))
    assert pkl_data_b is not None
    assert pkl_data_b.endswith("scene-B/data.pkl")
    assert pkl_data_a is not None
    assert pkl_data_a.endswith("scene-A/data.pkl")
    assert pkl_data_missing is None


@pytest.fixture
def restore_import_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo sys.path / sys.modules changes made while loading a foreign pickle."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        del sys.modules[name]


@pytest.mark.usefixtures("restore_import_state")
def test_unpickles_a_class_defined_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "utils").mkdir(parents=True)
    (repo / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "utils" / "scene.py").write_text(
        textwrap.dedent(
            """
            class IGScene:
                def __init__(self, data):
                    self.data = data
            """
        ),
        encoding="utf-8",
    )
    pickle_path = tmp_path / "data.pkl"
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import pickle, sys
                sys.path.insert(0, {str(repo)!r})
                from utils.scene import IGScene
                with open({str(pickle_path)!r}, "wb") as fh:
                    pickle.dump(IGScene({{"objs": []}}), fh)
                """
            ),
        ],
        check=True,
    )

    monkeypatch.chdir(repo)
    with pytest.raises(ModuleNotFoundError), pickle_path.open("rb") as handle:
        pickle.load(handle)

    assert dpc_runner.load_data_pkl(str(pickle_path)) == {"objs": []}
