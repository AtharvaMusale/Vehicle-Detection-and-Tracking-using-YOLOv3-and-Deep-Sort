"""Count each tracked vehicle once, when it crosses its lane's counting line."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from vehicle_tracking.geometry import Point
from vehicle_tracking.lanes import Lane
from vehicle_tracking.types import TrackedVehicle


@dataclass(frozen=True)
class CountEvent:
    frame: int
    track_id: int
    lane: str
    label: str


class LaneCounter:
    """Unique-ID, line-crossing counter.

    The earlier notebook incremented a counter for every detection in every frame,
    so a single car visible for 100 frames counted as 100 cars. Here a track is
    counted at most once, and only if it actually crosses the tripwire of the
    lane it is travelling in.
    """

    def __init__(self, lanes: list[Lane]) -> None:
        if not lanes:
            raise ValueError("At least one lane is required")
        names = [lane.name for lane in lanes]
        if len(set(names)) != len(names):
            raise ValueError("Lane names must be unique")
        self.lanes = lanes
        self._previous: dict[int, Point] = {}
        self._counted: set[int] = set()
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.events: list[CountEvent] = []

    def update(self, frame_index: int, tracks: list[TrackedVehicle]) -> list[CountEvent]:
        new_events: list[CountEvent] = []
        for track in tracks:
            current = track.anchor
            previous = self._previous.get(track.track_id)
            self._previous[track.track_id] = current
            if previous is None or track.track_id in self._counted:
                continue
            for lane in self.lanes:
                if lane.crossed(previous, current):
                    self._counted.add(track.track_id)
                    self._counts[lane.name][track.label] += 1
                    event = CountEvent(frame_index, track.track_id, lane.name, track.label)
                    new_events.append(event)
                    break
        self.events.extend(new_events)
        return new_events

    def forget(self, active_ids: set[int]) -> None:
        """Drop state for tracks that no longer exist (bounded memory on long videos)."""
        for track_id in set(self._previous) - active_ids:
            del self._previous[track_id]

    def lane_total(self, lane: str) -> int:
        return sum(self._counts[lane].values()) if lane in self._counts else 0

    def lane_breakdown(self, lane: str) -> dict[str, int]:
        return dict(self._counts[lane]) if lane in self._counts else {}

    @property
    def total(self) -> int:
        return len(self._counted)

    def summary(self) -> dict[str, object]:
        return {
            "total": self.total,
            "lanes": {
                lane.name: {
                    "total": self.lane_total(lane.name),
                    "by_class": self.lane_breakdown(lane.name),
                }
                for lane in self.lanes
            },
        }
