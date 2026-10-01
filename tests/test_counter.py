import pytest

from vehicle_tracking.counter import LaneCounter
from vehicle_tracking.lanes import Lane
from vehicle_tracking.types import TrackedVehicle


def lane(name="A", y=50.0):
    return Lane(name, ((0, 0), (100, 0), (100, 100), (0, 100)), ((0, y), (100, y)))


def vehicle(track_id, bottom, label="car", x=50.0):
    return TrackedVehicle(track_id, (x - 10, bottom - 20, x + 10, bottom), label, 0.9)


def test_counts_once_on_crossing():
    counter = LaneCounter([lane()])
    for frame, bottom in enumerate([30, 40, 55, 70, 90]):
        counter.update(frame, [vehicle(1, bottom)])
    assert counter.total == 1
    assert counter.lane_breakdown("A") == {"car": 1}


def test_does_not_count_vehicle_that_never_crosses():
    counter = LaneCounter([lane()])
    for frame in range(100):
        counter.update(frame, [vehicle(1, 20)])  # visible for 100 frames, never crosses
    assert counter.total == 0


def test_jitter_around_line_counts_only_once():
    counter = LaneCounter([lane()])
    for frame, bottom in enumerate([45, 55, 45, 55, 45, 55]):
        counter.update(frame, [vehicle(1, bottom)])
    assert counter.total == 1


def test_per_lane_and_class_breakdown():
    left = lane("L", 50)
    right = Lane("R", ((100, 0), (200, 0), (200, 100), (100, 100)), ((100, 50), (200, 50)))
    counter = LaneCounter([left, right])
    for frame, bottom in enumerate([40, 60]):
        counter.update(
            frame,
            [vehicle(1, bottom, "car", x=50), vehicle(2, bottom, "truck", x=150)],
        )
    assert counter.lane_breakdown("L") == {"car": 1}
    assert counter.lane_breakdown("R") == {"truck": 1}
    assert counter.summary()["total"] == 2
    assert [e.track_id for e in counter.events] == [1, 2]


def test_forget_bounds_memory_but_keeps_counts():
    counter = LaneCounter([lane()])
    counter.update(0, [vehicle(1, 40)])
    counter.update(1, [vehicle(1, 60)])
    counter.forget(set())
    assert counter.total == 1
    assert not counter._previous


def test_validation():
    with pytest.raises(ValueError):
        LaneCounter([])
    with pytest.raises(ValueError):
        LaneCounter([lane("A"), lane("A")])
