import json
from pathlib import Path

import pytest

TRAINING = Path(__file__).resolve().parents[2] / "training"


def test_training_gate_fails_closed(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING))
    from training_gate import require_training_approval
    with pytest.raises(RuntimeError, match="no data-quality"):
        require_training_approval(tmp_path)
    (tmp_path / "state").mkdir()
    gate = tmp_path / "state" / "training_gate.json"
    gate.write_text(json.dumps({"training_approved": False, "reason": "bad geometry"}))
    with pytest.raises(RuntimeError, match="bad geometry"):
        require_training_approval(tmp_path)
    require_training_approval(tmp_path, smoke=True)
    gate.write_text(json.dumps({"training_approved": True}))
    require_training_approval(tmp_path)


def test_detector_reads_the_run_length_masks_of_the_generator():
    trainer = (TRAINING / "train_detector.py").read_text()
    assert 'cfg.INPUT.MASK_FORMAT = "bitmask"' in trainer
