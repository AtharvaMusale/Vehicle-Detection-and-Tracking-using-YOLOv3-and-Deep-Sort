"""Typed configuration loaded from YAML.

Lane geometry is stored in normalised (0-1) coordinates so a config keeps working
if the video resolution changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vehicle_tracking.geometry import Point
from vehicle_tracking.lanes import Lane

_DEFAULT_COLORS = [(255, 170, 0), (0, 200, 120), (60, 90, 255), (200, 80, 220)]  # BGR


@dataclass(frozen=True)
class DetectorConfig:
    cfg_path: Path = Path("models/yolov3.cfg")
    weights_path: Path = Path("models/yolov3.weights")
    names_path: Path | None = None  # defaults to the bundled COCO names
    input_size: int = 416
    score_threshold: float = 0.3
    nms_threshold: float = 0.45
    classes: tuple[str, ...] = ("car", "bus", "truck", "motorbike")


@dataclass(frozen=True)
class TrackerConfig:
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.4


@dataclass(frozen=True)
class VideoConfig:
    process_width: int = 1280
    detect_every: int = 1
    start_seconds: float = 0.0
    max_frames: int | None = None


@dataclass(frozen=True)
class RenderConfig:
    trail_length: int = 30
    lane_opacity: float = 0.18
    show_hud: bool = True


@dataclass(frozen=True)
class LaneConfig:
    name: str
    polygon: tuple[Point, ...]
    line: tuple[Point, Point]
    color: tuple[int, int, int] | None = None

    def to_lane(self, width: int, height: int, index: int = 0) -> Lane:
        def scale(p: Point) -> Point:
            return (p[0] * width, p[1] * height)

        return Lane(
            name=self.name,
            polygon=tuple(scale(p) for p in self.polygon),
            line=(scale(self.line[0]), scale(self.line[1])),
            color=self.color or _DEFAULT_COLORS[index % len(_DEFAULT_COLORS)],
        )


@dataclass(frozen=True)
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    lanes: tuple[LaneConfig, ...] = ()
    roi: tuple[Point, ...] | None = None  # only vehicles standing inside are tracked

    def build_roi(self, width: int, height: int) -> tuple[Point, ...] | None:
        if self.roi is None:
            return None
        return tuple((x * width, y * height) for x, y in self.roi)

    def build_lanes(self, width: int, height: int) -> list[Lane]:
        return [lane.to_lane(width, height, i) for i, lane in enumerate(self.lanes)]


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return value


def _points(values: Any, minimum: int, what: str) -> tuple[Point, ...]:
    pts = tuple((float(x), float(y)) for x, y in values)
    if len(pts) < minimum:
        raise ValueError(f"{what} needs at least {minimum} points")
    for x, y in pts:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{what} points must be normalised to [0, 1], got {(x, y)}")
    return pts


def parse_config(raw: dict[str, Any], base_dir: Path | None = None) -> Config:
    base = base_dir or Path.cwd()

    det = _section(raw, "detector")
    detector = DetectorConfig(
        cfg_path=base / det.get("cfg_path", "models/yolov3.cfg"),
        weights_path=base / det.get("weights_path", "models/yolov3.weights"),
        names_path=(base / det["names_path"]) if det.get("names_path") else None,
        input_size=int(det.get("input_size", 416)),
        score_threshold=float(det.get("score_threshold", 0.3)),
        nms_threshold=float(det.get("nms_threshold", 0.45)),
        classes=tuple(det.get("classes", DetectorConfig.classes)),
    )
    if detector.input_size % 32:
        raise ValueError("detector.input_size must be a multiple of 32")

    trk = _section(raw, "tracker")
    tracker = TrackerConfig(**{k: trk[k] for k in trk if k in TrackerConfig.__dataclass_fields__})

    vid = _section(raw, "video")
    video = VideoConfig(**{k: vid[k] for k in vid if k in VideoConfig.__dataclass_fields__})
    if video.detect_every < 1:
        raise ValueError("video.detect_every must be >= 1")

    ren = _section(raw, "render")
    render = RenderConfig(**{k: ren[k] for k in ren if k in RenderConfig.__dataclass_fields__})

    lanes = []
    for entry in raw.get("lanes") or []:
        line = _points(entry["line"], 2, f"lane '{entry.get('name')}' line")
        if len(line) != 2:
            raise ValueError("lane line must have exactly 2 points")
        color = tuple(int(c) for c in entry["color"]) if "color" in entry else None
        lanes.append(
            LaneConfig(
                name=str(entry["name"]),
                polygon=_points(entry["polygon"], 3, f"lane '{entry['name']}' polygon"),
                line=(line[0], line[1]),
                color=color,  # type: ignore[arg-type]
            )
        )
    roi = _points(raw["roi"], 3, "roi") if raw.get("roi") else None
    return Config(detector, tracker, video, render, tuple(lanes), roi)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    return parse_config(raw, base_dir=path.resolve().parent.parent)
