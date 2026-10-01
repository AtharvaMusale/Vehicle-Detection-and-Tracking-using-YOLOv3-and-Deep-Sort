"""Lane definitions: a region of interest plus a counting line."""

from __future__ import annotations

from dataclasses import dataclass

from vehicle_tracking.geometry import Point, point_in_polygon, segments_intersect


@dataclass(frozen=True)
class Lane:
    """A lane in pixel coordinates.

    ``polygon`` decides which lane a vehicle belongs to; ``line`` is the virtual
    tripwire a vehicle must cross to be counted.
    """

    name: str
    polygon: tuple[Point, ...]
    line: tuple[Point, Point]
    color: tuple[int, int, int] = (0, 200, 255)  # BGR

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    def crossed(self, previous: Point, current: Point) -> bool:
        return segments_intersect(previous, current, self.line[0], self.line[1])


def find_lane(lanes: list[Lane], point: Point) -> Lane | None:
    """First lane whose polygon contains the point."""
    for lane in lanes:
        if lane.contains(point):
            return lane
    return None
