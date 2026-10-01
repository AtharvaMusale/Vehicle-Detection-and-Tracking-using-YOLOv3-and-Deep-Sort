from vehicle_tracking.geometry import point_in_polygon, segments_intersect

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_inside_and_outside():
    assert point_in_polygon((5, 5), SQUARE)
    assert not point_in_polygon((15, 5), SQUARE)


def test_point_on_edge_counts_as_inside():
    assert point_in_polygon((10, 5), SQUARE)
    assert point_in_polygon((0, 0), SQUARE)


def test_concave_polygon():
    poly = [(0, 0), (10, 0), (10, 10), (5, 4), (0, 10)]
    assert point_in_polygon((2, 2), poly)
    assert not point_in_polygon((5, 8), poly)


def test_segment_crossing():
    assert segments_intersect((5, 0), (5, 10), (0, 5), (10, 5))
    assert not segments_intersect((5, 0), (5, 4), (0, 5), (10, 5))
    # crosses the infinite line but outside the finite segment
    assert not segments_intersect((20, 0), (20, 10), (0, 5), (10, 5))
