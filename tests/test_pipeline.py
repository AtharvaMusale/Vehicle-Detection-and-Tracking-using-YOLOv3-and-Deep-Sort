"""End-to-end test on a synthetic video with a fake detector (no model files needed)."""

import json

import cv2
import numpy as np
import pytest

from vehicle_tracking.config import parse_config
from vehicle_tracking.pipeline import run_pipeline
from vehicle_tracking.types import Detection

W, H, FRAMES = 320, 240, 40


class MovingBoxDetector:
    """Reports the rectangle drawn by ``make_video`` (ground truth)."""

    def __init__(self):
        self.frame_index = 0

    def detect(self, frame):
        y = 20 + self.frame_index * 5
        self.frame_index += 1
        return [Detection((140.0, float(y), 40.0, 30.0), 0.95, "car")]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
    for i in range(FRAMES):
        frame = np.full((H, W, 3), 60, dtype=np.uint8)
        y = 20 + i * 5
        cv2.rectangle(frame, (140, y), (180, y + 30), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return path


CONFIG = {
    "video": {"process_width": 320},
    "tracker": {"n_init": 2},
    "lanes": [
        {
            "name": "Only lane",
            "polygon": [[0.2, 0.0], [0.8, 0.0], [0.8, 1.0], [0.2, 1.0]],
            "line": [[0.2, 0.6], [0.8, 0.6]],
        }
    ],
}


def test_counts_single_vehicle_once_and_writes_outputs(video, tmp_path):
    out = tmp_path / "out" / "demo.mp4"
    summary = run_pipeline(parse_config(CONFIG), MovingBoxDetector(), video, out)
    assert summary["total"] == 1
    assert summary["frames"] == FRAMES
    assert summary["lanes"]["Only lane"]["by_class"] == {"car": 1}
    assert out.is_file() and out.stat().st_size > 0
    assert json.loads(out.with_suffix(".summary.json").read_text())["total"] == 1
    assert len(out.with_suffix(".events.csv").read_text().strip().splitlines()) == 2
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == FRAMES


def test_max_frames_limits_processing(video, tmp_path):
    config = parse_config({**CONFIG, "video": {"process_width": 320, "max_frames": 5}})
    summary = run_pipeline(config, MovingBoxDetector(), video, tmp_path / "o.mp4")
    assert summary["frames"] == 5


def test_requires_lanes(video, tmp_path):
    with pytest.raises(ValueError, match="no lanes"):
        run_pipeline(parse_config({}), MovingBoxDetector(), video, tmp_path / "o.mp4")


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            parse_config(CONFIG), MovingBoxDetector(), tmp_path / "nope.mp4", tmp_path / "o.mp4"
        )


def test_roi_excludes_detections_outside(video, tmp_path):
    # ROI covers only the bottom third, so the box never enters it before the tripwire
    raw = {**CONFIG, "roi": [[0.0, 0.95], [1.0, 0.95], [1.0, 1.0], [0.0, 1.0]]}
    summary = run_pipeline(parse_config(raw), MovingBoxDetector(), video, tmp_path / "o.mp4")
    assert summary["total"] == 0
