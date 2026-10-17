import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

RUNNERS = Path(__file__).resolve().parents[1] / "runners"
sys.path.insert(0, str(RUNNERS))
import dpc_runner  # noqa: E402


class DpcConvertTest(unittest.TestCase):
    def test_convert_data_pkl(self) -> None:
        data = {
            "objs": [
                {
                    "classname": "chair",
                    "score": 0.9,
                    "bdb3d": {
                        "centroid": [0, 0, 0],
                        "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "size": [2, 2, 2],
                    },
                },
                {"classname": "no-3d-box"},
            ],
            "layout": {
                "manhattan_world": [
                    [2, 3, 4], [2, 3, -4], [2, -3, 4], [2, -3, -4],
                    [-2, 3, 4], [-2, 3, -4], [-2, -3, 4], [-2, -3, -4],
                ],
            },
        }
        scene = dpc_runner.convert_data_pkl(data)
        self.assertEqual(len(scene["objects"]), 1)
        self.assertEqual(scene["objects"][0]["label"], "chair")
        self.assertEqual(scene["objects"][0]["score"], 0.9)
        bbox = scene["objects"][0]["bbox"]
        self.assertEqual(bbox["center"], [0.0, 0.0, 0.0])
        self.assertEqual(bbox["size"], [2.0, 2.0, 2.0])
        self.assertEqual(bbox["basis"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(scene["layout"]["min_corner"], [-2.0, -3.0, -4.0])
        self.assertEqual(scene["layout"]["max_corner"], [2.0, 3.0, 4.0])

    def test_a_rotated_box_keeps_its_corners_through_the_contract(self) -> None:
        import math
        c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
        dpc_basis = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        centroid, size = [1.0, 2.0, 0.5], [2.0, 1.0, 0.8]
        data = {"objs": [{"classname": "table", "bdb3d": {
            "centroid": centroid, "basis": dpc_basis, "size": size}}]}
        signs = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
        dpc_corners = [[centroid[i] + sum(dpc_basis[i][k] * signs_k * size[k]
                                          for k, signs_k in enumerate(sign)) for i in range(3)]
                       for sign in signs]
        bbox = dpc_runner.convert_data_pkl(data)["objects"][0]["bbox"]
        rows = bbox["basis"]
        ours = [[bbox["center"][i] + sum(sign[k] * bbox["size"][k] * rows[k][i] for k in range(3))
                 for i in range(3)] for sign in signs]
        worst = max(min(math.dist(a, b) for b in ours) for a in dpc_corners)
        self.assertLess(worst, 1e-9)

    def test_boxes_and_layout_are_centred_on_the_camera(self) -> None:
        data = {
            "camera": {"pos": [10.0, 20.0, 1.5]},
            "objs": [{"classname": "bed", "bdb3d": {
                "centroid": [11.0, 22.0, 0.5], "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "size": [2, 1, 1]}}],
            "layout": {"manhattan_world": [[8, 18, 0], [12, 18, 0], [12, 23, 0], [8, 23, 0],
                                           [8, 18, 2.6], [12, 18, 2.6], [12, 23, 2.6], [8, 23, 2.6]]},
        }
        scene = dpc_runner.convert_data_pkl(data)
        self.assertEqual(scene["objects"][0]["bbox"]["center"], [1.0, 2.0, -1.0])
        self.assertEqual(scene["layout"]["min_corner"], [-2.0, -2.0, -1.5])
        self.assertEqual(scene["layout"]["max_corner"], [2.0, 3.0, 1.1])

    def test_a_front3d_sample_finds_the_folder_dpc_wrote(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            pkl = Path(root) / "run1" / "visualization" / "house-1" / "Bedroom-2__3" / "data.pkl"
            pkl.parent.mkdir(parents=True)
            pkl.write_bytes(b"")
            found = dpc_runner.find_data_pkl("house-1/Bedroom-2/3", search_root=root)
            self.assertTrue(found.endswith("house-1/Bedroom-2__3/data.pkl"))

    def test_convert_attaches_mesh_path_as_shape(self) -> None:
        data = {
            "objs": [
                {
                    "classname": "chair",
                    "mesh_path": "/abs/pred/s1/obj-0.ply",
                    "bdb3d": {
                        "centroid": [0, 0, 0],
                        "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "size": [2, 2, 2],
                    },
                },
                {
                    "classname": "table",
                    "mesh": object(),
                    "bdb3d": {
                        "centroid": [5, 5, 5],
                        "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "size": [1, 1, 1],
                    },
                },
            ],
            "layout": None,
        }
        scene = dpc_runner.convert_data_pkl(data)
        self.assertEqual(
            scene["objects"][0]["attributes"]["shape"], "/abs/pred/s1/obj-0.ply"
        )
        self.assertNotIn("shape", scene["objects"][1]["attributes"])

    def test_find_data_pkl_matches_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            for scene in ("scene-A", "scene-B"):
                pkl = Path(root) / "run1" / "visualization" / scene / "data.pkl"
                pkl.parent.mkdir(parents=True, exist_ok=True)
                pkl.write_bytes(b"")
            found_b = dpc_runner.find_data_pkl("scene-B", search_root=root)
            found_a = dpc_runner.find_data_pkl("scene-A", search_root=root)
            self.assertTrue(found_b.endswith("scene-B/data.pkl"))
            self.assertTrue(found_a.endswith("scene-A/data.pkl"))
            self.assertIsNone(dpc_runner.find_data_pkl("missing", search_root=root))


class LoadDataPklTest(unittest.TestCase):
    def test_unpickles_a_class_defined_in_the_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
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
            pickle_path = Path(root) / "data.pkl"
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

            previous_cwd = Path.cwd()
            previous_path = list(sys.path)
            previous_modules = set(sys.modules)
            try:
                os.chdir(repo)
                with self.assertRaises(ModuleNotFoundError):
                    with open(pickle_path, "rb") as handle:
                        pickle.load(handle)
                self.assertEqual(dpc_runner.load_data_pkl(str(pickle_path)), {"objs": []})
            finally:
                os.chdir(previous_cwd)
                sys.path[:] = previous_path
                for name in set(sys.modules) - previous_modules:
                    del sys.modules[name]


if __name__ == "__main__":
    unittest.main()
