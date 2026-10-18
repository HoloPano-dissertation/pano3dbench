from pano3dbench.domain import BoundingBox3D, SceneOutput


def test_scene_output_round_trip() -> None:
    original = SceneOutput(layout=BoundingBox3D((0, 0, 0), (1, 2, 3)))
    assert SceneOutput.from_dict(original.to_dict()) == original
