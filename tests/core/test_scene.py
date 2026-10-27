from pano3dbench.geometry.boxes import BoundingBox3D
from pano3dbench.scene import SceneOutput


def test_scene_output_round_trip() -> None:
    original = SceneOutput(layout=BoundingBox3D((0, 0, 0), (1, 2, 3)))
    assert SceneOutput.from_dict(original.to_dict()) == original
