from pathlib import Path

import pytest

from vehicle_tracking.config import load_config, parse_config

ROOT = Path(__file__).resolve().parent.parent


def test_default_config_loads_and_scales_lanes():
    config = load_config(ROOT / "configs" / "default.yaml")
    assert config.detector.classes == ("car", "bus", "truck", "motorbike")
    lanes = config.build_lanes(1000, 500)
    assert len(lanes) == len(config.lanes) >= 1
    assert all(0 <= x <= 1000 and 0 <= y <= 500 for lane in lanes for x, y in lane.polygon)
    assert lanes[0].color != lanes[1].color


def test_rejects_unnormalised_points():
    raw = {"lanes": [{"name": "x", "polygon": [[0, 0], [5, 0], [1, 1]], "line": [[0, 0], [1, 1]]}]}
    with pytest.raises(ValueError, match="normalised"):
        parse_config(raw)


def test_rejects_bad_input_size():
    with pytest.raises(ValueError, match="multiple of 32"):
        parse_config({"detector": {"input_size": 400}})


def test_rejects_line_with_wrong_point_count():
    raw = {
        "lanes": [
            {"name": "x", "polygon": [[0, 0], [1, 0], [1, 1]], "line": [[0, 0], [1, 1], [0, 1]]}
        ]
    }
    with pytest.raises(ValueError, match="exactly 2"):
        parse_config(raw)


def test_roi_parsed_and_scaled():
    cfg = parse_config({"roi": [[0, 0], [1, 0], [1, 1]]})
    assert cfg.build_roi(200, 100) == ((0.0, 0.0), (200.0, 0.0), (200.0, 100.0))
    assert parse_config({}).build_roi(200, 100) is None
