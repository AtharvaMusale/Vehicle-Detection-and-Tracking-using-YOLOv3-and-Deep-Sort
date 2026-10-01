"""End-to-end pipeline: read video -> detect -> track -> count -> render -> write."""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from vehicle_tracking.config import Config
from vehicle_tracking.counter import LaneCounter
from vehicle_tracking.geometry import point_in_polygon
from vehicle_tracking.tracker import VehicleTracker
from vehicle_tracking.types import Array, Detection
from vehicle_tracking.video_io import VideoReader, VideoWriter
from vehicle_tracking.visualize import Renderer

logger = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, frame: Array) -> list[Detection]: ...


def run_pipeline(
    config: Config,
    detector: Detector,
    video_path: Path,
    output_path: Path,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Process a video and return the counting summary.

    Also writes ``<output>.summary.json`` and ``<output>.events.csv`` next to the video.
    """
    if not config.lanes:
        raise ValueError("Config defines no lanes; add at least one under 'lanes:'")
    reader = VideoReader(video_path, config.video.process_width, config.video.start_seconds)
    lanes = config.build_lanes(reader.width, reader.height)
    roi = config.build_roi(reader.width, reader.height)
    counter = LaneCounter(lanes)
    tracker = VehicleTracker(config.tracker)
    renderer = Renderer(config.render, lanes)

    limit = config.video.max_frames
    total = min(reader.total_frames, limit) if limit else reader.total_frames
    detections: list[Detection] = []
    started = time.perf_counter()
    processed = 0
    try:
        with VideoWriter(output_path, reader.width, reader.height, reader.fps) as writer:
            for index, frame in enumerate(reader):
                if limit and index >= limit:
                    break
                if index % config.video.detect_every == 0:
                    detections = _inside_roi(detector.detect(frame), roi)
                tracks = tracker.update(
                    frame, detections if index % config.video.detect_every == 0 else []
                )
                events = counter.update(index, tracks)
                counter.forget({t.track_id for t in tracks})
                elapsed = time.perf_counter() - started
                fps = (processed + 1) / elapsed if elapsed > 0 else None
                writer.write(renderer.draw(frame, tracks, counter, events, fps))
                processed += 1
                if progress:
                    progress(processed, total)
    finally:
        reader.release()

    elapsed = time.perf_counter() - started
    summary = counter.summary()
    summary.update(
        frames=processed,
        seconds=round(elapsed, 2),
        processing_fps=round(processed / elapsed, 2) if elapsed else None,
        video=str(video_path),
    )
    _write_reports(output_path, summary, counter)
    return summary


def _inside_roi(
    detections: list[Detection], roi: tuple[tuple[float, float], ...] | None
) -> list[Detection]:
    """Keep detections whose bottom-centre (road contact point) lies in the ROI."""
    if roi is None:
        return detections
    kept = []
    for det in detections:
        left, top, width, height = det.ltwh
        if point_in_polygon((left + width / 2, top + height), roi):
            kept.append(det)
    return kept


def _write_reports(output_path: Path, summary: dict[str, object], counter: LaneCounter) -> None:
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    with output_path.with_suffix(".events.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "track_id", "lane", "class"])
        for e in counter.events:
            writer.writerow([e.frame, e.track_id, e.lane, e.label])
