"""Deep SORT tracker wrapper producing confirmed ``TrackedVehicle`` objects."""

from __future__ import annotations

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from vehicle_tracking.config import TrackerConfig
from vehicle_tracking.encoder import ColorHistogramEncoder
from vehicle_tracking.types import Detection, TrackedVehicle


class VehicleTracker:
    def __init__(self, config: TrackerConfig, encoder: ColorHistogramEncoder | None = None) -> None:
        self._encoder = encoder or ColorHistogramEncoder()
        self._deepsort = DeepSort(
            max_age=config.max_age,
            n_init=config.n_init,
            max_iou_distance=config.max_iou_distance,
            max_cosine_distance=config.max_cosine_distance,
            embedder=None,
        )
        self._labels: dict[int, str] = {}

    def update(self, frame: np.ndarray, detections: list[Detection]) -> list[TrackedVehicle]:
        embeds = self._encoder.encode(frame, detections)
        raw = [(list(d.ltwh), d.score, d.label) for d in detections]
        tracks = self._deepsort.update_tracks(raw, embeds=embeds, frame=frame)
        active: list[TrackedVehicle] = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = int(track.track_id)
            label = track.get_det_class() or self._labels.get(track_id, "vehicle")
            self._labels[track_id] = label
            score = track.get_det_conf()
            ltrb = tuple(float(v) for v in track.to_ltrb())
            active.append(
                TrackedVehicle(
                    track_id=track_id,
                    ltrb=ltrb,  # type: ignore[arg-type]
                    label=label,
                    score=float(score) if score is not None else 0.0,
                )
            )
        return active
