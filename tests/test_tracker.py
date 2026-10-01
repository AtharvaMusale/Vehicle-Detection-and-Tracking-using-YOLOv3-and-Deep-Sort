import numpy as np

from vehicle_tracking.config import TrackerConfig
from vehicle_tracking.tracker import VehicleTracker
from vehicle_tracking.types import Detection


def test_label_is_majority_vote_not_latest():
    tracker = VehicleTracker(TrackerConfig(n_init=1))
    frame = np.full((300, 400, 3), 90, dtype=np.uint8)
    labels = ["car", "car", "car", "bus", "car"]  # one noisy frame
    seen = []
    for i, label in enumerate(labels):
        out = tracker.update(frame, [Detection((100.0 + i * 4, 100.0, 60.0, 40.0), 0.9, label)])
        seen.extend(t.label for t in out)
    assert seen and set(seen) == {"car"}


def test_track_id_is_stable_for_moving_object():
    tracker = VehicleTracker(TrackerConfig(n_init=2))
    frame = np.full((300, 400, 3), 90, dtype=np.uint8)
    ids = set()
    for i in range(15):
        for t in tracker.update(frame, [Detection((50.0 + i * 5, 100.0, 60.0, 40.0), 0.9, "car")]):
            ids.add(t.track_id)
    assert len(ids) == 1
