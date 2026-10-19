import importlib.util
import json

import pytest

pytest.importorskip("front3d_pano")

np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def metadata():
    return importlib.import_module("complete_dpc_metadata")


def test_each_class_keeps_its_own_average(metadata):
    sizes = {"bed": [np.array([2.0, 1.0, 0.5]), np.array([2.0, 3.0, 0.5])],
             "chair": [np.array([0.5, 0.5, 1.0])]}
    averaged = metadata.average_sizes(sizes, ["bed", "chair"])
    assert np.allclose(averaged[0], [2.0, 2.0, 0.5])
    assert np.allclose(averaged[1], [0.5, 0.5, 1.0])


def test_a_class_without_objects_gets_the_overall_average(metadata):
    sizes = {"bed": [np.array([2.0, 2.0, 2.0])], "chair": [np.array([1.0, 1.0, 1.0])]}
    averaged = metadata.average_sizes(sizes, ["bed", "chair", "appliance"])
    assert averaged.shape == (3, 3)
    assert np.allclose(averaged[2], [1.5, 1.5, 1.5])


def test_the_order_follows_the_class_list_not_the_data(metadata):
    sizes = {"chair": [np.array([1.0, 1.0, 1.0])], "bed": [np.array([2.0, 2.0, 2.0])]}
    averaged = metadata.average_sizes(sizes, ["bed", "chair"])
    assert np.allclose(averaged[0], [2.0, 2.0, 2.0])


def test_an_empty_set_is_refused_rather_than_averaged_to_nothing(metadata):
    with pytest.raises(ValueError, match="No object sizes"):
        metadata.average_sizes({}, ["bed"])


def test_collect_reads_every_scene_and_keeps_the_farthest(metadata, tmp_path):
    import pickle
    root = tmp_path / "scenes"
    folders = []
    for index, (size, distance) in enumerate(((1.0, 3.0), (3.0, 7.0))):
        folder = root / "train" / f"room{index}"
        folder.mkdir(parents=True)
        with (folder / "data.pkl").open("wb") as handle:
            pickle.dump({"objs": [{"classname": "bed",
                                   "bdb3d": {"size": [size, size, size],
                                             "dis": distance}}]}, handle)
        folders.append(str(folder.relative_to(root)))
    (root / "train.json").write_text(json.dumps(folders))

    class Transform:
        @staticmethod
        def world2campix(box):
            return {"dis": box["dis"]}

    sizes, farthest = metadata.collect(root, "train", lambda scene: Transform())
    assert farthest == 7.0
    assert len(sizes["bed"]) == 2
