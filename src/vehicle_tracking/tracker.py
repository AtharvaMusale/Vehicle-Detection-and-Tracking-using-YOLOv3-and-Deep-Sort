"""Deep SORT tracker wrapper producing confirmed ``TrackedVehicle`` objects."""

from __future__ import annotations

from collections import Counter, defaultdict

from deep_sort_realtime.deepsort_tracker import DeepSort

from vehicle_tracking.config import TrackerConfig
from vehicle_tracking.encoder import ColorHistogramEncoder
from vehicle_tracking.types import Array, Detection, TrackedVehicle


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
        self._votes: dict[int, Counter[str]] = defaultdict(Counter)

    def update(self, frame: Array, detections: list[Detection]) -> list[TrackedVehicle]:
        embeds = self._encoder.encode(frame, detections)
        raw = [(list(d.ltwh), d.score, d.label) for d in detections]
        tracks = self._deepsort.update_tracks(raw, embeds=embeds, frame=frame)
        active: list[TrackedVehicle] = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = int(track.track_id)
            latest = track.get_det_class()
            if latest:
                self._votes[track_id][latest] += 1
            # Majority vote over the track's lifetime avoids car/truck/bus flicker.
            votes = self._votes[track_id]
            label = votes.most_common(1)[0][0] if votes else "vehicle"
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
