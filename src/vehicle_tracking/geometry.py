"""Small 2-D geometry helpers (no OpenCV dependency, easy to unit test)."""

from __future__ import annotations

from collections.abc import Sequence

Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting test; points exactly on an edge count as inside."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if _on_segment(point, (x1, y1), (x2, y2)):
            return True
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def _on_segment(p: Point, a: Point, b: Point, eps: float = 1e-9) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if abs(cross) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def side_of_line(point: Point, a: Point, b: Point) -> float:
    """Signed area: >0 left of a->b, <0 right of a->b, 0 on the line."""
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if the movement p1->p2 crosses the finite segment q1->q2."""
    d1 = side_of_line(p1, q1, q2)
    d2 = side_of_line(p2, q1, q2)
    d3 = side_of_line(q1, p1, p2)
    d4 = side_of_line(q2, p1, p2)
    return d1 * d2 < 0 and d3 * d4 < 0
