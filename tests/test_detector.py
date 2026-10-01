"""Detector post-processing tested with synthetic network output (no weights needed)."""

import numpy as np
import pytest

from vehicle_tracking.config import DetectorConfig
from vehicle_tracking.detector import YoloV3Detector, load_class_names


def test_bundled_class_names():
    names = load_class_names()
    assert len(names) == 80 and names[2] == "car" and names[7] == "truck"


def test_missing_model_files_give_actionable_error(tmp_path):
    cfg = DetectorConfig(cfg_path=tmp_path / "a.cfg", weights_path=tmp_path / "a.weights")
    with pytest.raises(FileNotFoundError, match="download-assets"):
        YoloV3Detector(cfg)


def _detector_without_network():
    det = object.__new__(YoloV3Detector)
    det.config = DetectorConfig()
    det.names = load_class_names()
    det._wanted = {2, 5, 7, 3}
    return det


def _row(cx, cy, w, h, obj, class_id):
    row = np.zeros(85, dtype=np.float32)
    row[:4] = (cx, cy, w, h)
    row[4] = obj
    row[5 + class_id] = 1.0
    return row


def test_postprocess_filters_classes_scores_and_overlaps():
    det = _detector_without_network()
    output = np.stack(
        [
            _row(0.5, 0.5, 0.2, 0.2, 0.9, 2),  # car, kept
            _row(0.5, 0.5, 0.2, 0.2, 0.8, 2),  # duplicate of the car, removed by NMS
            _row(0.2, 0.2, 0.1, 0.1, 0.9, 0),  # person, filtered out
            _row(0.8, 0.8, 0.1, 0.1, 0.1, 7),  # truck below threshold
            _row(0.8, 0.2, 0.1, 0.1, 0.7, 7),  # truck, kept
        ]
    )
    result = det._postprocess([output], 1000, 500)
    assert sorted(d.label for d in result) == ["car", "truck"]
    car = next(d for d in result if d.label == "car")
    assert car.ltwh == pytest.approx((400.0, 200.0, 200.0, 100.0), abs=1.5)
    assert det._postprocess([np.zeros((0, 85), dtype=np.float32)], 1000, 500) == []
