"""Rendering of the demo overlay: lanes, tripwires, boxes, trails and a stats HUD."""

from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np

from vehicle_tracking.config import RenderConfig
from vehicle_tracking.counter import CountEvent, LaneCounter
from vehicle_tracking.lanes import Lane
from vehicle_tracking.types import TrackedVehicle

FONT = cv2.FONT_HERSHEY_DUPLEX
CLASS_COLORS = {  # BGR
    "car": (255, 180, 40),
    "bus": (60, 90, 255),
    "truck": (40, 200, 255),
    "motorbike": (200, 90, 230),
}
DEFAULT_COLOR = (230, 230, 230)
FLASH_FRAMES = 12


def _ipoint(p: tuple[float, float]) -> tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


class Renderer:
    def __init__(self, config: RenderConfig, lanes: list[Lane]) -> None:
        self.config = config
        self.lanes = lanes
        self._trails: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=max(config.trail_length, 2))
        )
        self._flash: dict[str, int] = {}
        self._lane_layer: np.ndarray | None = None
        self._lane_mask: np.ndarray | None = None

    # -- public -----------------------------------------------------------------
    def draw(
        self,
        frame: np.ndarray,
        tracks: list[TrackedVehicle],
        counter: LaneCounter,
        events: list[CountEvent],
        fps: float | None = None,
    ) -> np.ndarray:
        for event in events:
            self._flash[event.lane] = FLASH_FRAMES
        out = self._draw_lanes(frame)
        self._draw_trails(out, tracks)
        for track in tracks:
            self._draw_vehicle(out, track)
        if self.config.show_hud:
            self._draw_hud(out, counter, fps)
        self._flash = {k: v - 1 for k, v in self._flash.items() if v > 1}
        return out

    # -- lanes ------------------------------------------------------------------
    def _draw_lanes(self, frame: np.ndarray) -> np.ndarray:
        if self._lane_layer is None or self._lane_layer.shape != frame.shape:
            layer = np.zeros_like(frame)
            for lane in self.lanes:
                pts = np.array([_ipoint(p) for p in lane.polygon], dtype=np.int32)
                cv2.fillPoly(layer, [pts], lane.color)
            self._lane_layer = layer
            self._lane_mask = np.asarray(layer.any(axis=2))
        mask = self._lane_mask
        out: np.ndarray = frame.copy()
        alpha = self.config.lane_opacity
        blended: np.ndarray = cv2.addWeighted(frame, 1 - alpha, self._lane_layer, alpha, 0)
        out[mask] = blended[mask]
        for lane in self.lanes:
            pts = np.array([_ipoint(p) for p in lane.polygon], dtype=np.int32)
            cv2.polylines(out, [pts], True, lane.color, 1, cv2.LINE_AA)
            flashing = self._flash.get(lane.name, 0)
            color = (255, 255, 255) if flashing else lane.color
            thickness = 6 if flashing else 3
            cv2.line(
                out, _ipoint(lane.line[0]), _ipoint(lane.line[1]), color, thickness, cv2.LINE_AA
            )
        return out

    # -- vehicles ---------------------------------------------------------------
    def _draw_trails(self, frame: np.ndarray, tracks: list[TrackedVehicle]) -> None:
        live = {t.track_id for t in tracks}
        for track in tracks:
            self._trails[track.track_id].append(_ipoint(track.anchor))
        for stale in set(self._trails) - live:
            del self._trails[stale]
        for track in tracks:
            pts = list(self._trails[track.track_id])
            color = CLASS_COLORS.get(track.label, DEFAULT_COLOR)
            for i in range(1, len(pts)):
                fade = i / len(pts)
                cv2.line(frame, pts[i - 1], pts[i], color, max(1, int(3 * fade)), cv2.LINE_AA)

    def _draw_vehicle(self, frame: np.ndarray, track: TrackedVehicle) -> None:
        color = CLASS_COLORS.get(track.label, DEFAULT_COLOR)
        x1, y1, x2, y2 = (int(v) for v in track.ltrb)
        arm = max(8, min(x2 - x1, y2 - y1) // 4)
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, 2, cv2.LINE_AA)
        text = f"{track.label.upper()} #{track.track_id}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        top = max(y1 - th - 10, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 10, top + th + 8), color, -1)
        cv2.putText(frame, text, (x1 + 5, top + th + 2), FONT, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    # -- HUD --------------------------------------------------------------------
    def _draw_hud(self, frame: np.ndarray, counter: LaneCounter, fps: float | None) -> None:
        row_h = 26
        width = 280
        height = 70 + row_h * len(self.lanes) + 22
        x0, y0 = 16, 16
        panel = frame.copy()
        cv2.rectangle(panel, (x0, y0), (x0 + width, y0 + height), (24, 24, 28), -1)
        cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)
        cv2.putText(
            frame,
            "VEHICLES COUNTED",
            (x0 + 14, y0 + 24),
            FONT,
            0.5,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(counter.total),
            (x0 + 14, y0 + 62),
            FONT,
            1.3,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y = y0 + 90
        for lane in self.lanes:
            cv2.rectangle(frame, (x0 + 14, y - 12), (x0 + 24, y - 2), lane.color, -1)
            cv2.putText(
                frame, f"{lane.name}", (x0 + 32, y), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA
            )
            value = str(counter.lane_total(lane.name))
            (vw, _), _ = cv2.getTextSize(value, FONT, 0.55, 1)
            cv2.putText(
                frame, value, (x0 + width - 14 - vw, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )
            y += row_h
        classes: dict[str, int] = defaultdict(int)
        for lane in self.lanes:
            for label, n in counter.lane_breakdown(lane.name).items():
                classes[label] += n
        summary = "  ".join(f"{k} {v}" for k, v in sorted(classes.items())) or "no vehicles yet"
        cv2.putText(
            frame, summary, (x0 + 14, y0 + height - 10), FONT, 0.4, (160, 160, 160), 1, cv2.LINE_AA
        )
        if fps is not None:
            label = f"{fps:.1f} FPS"
            (fw, fh), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            h_img, w_img = frame.shape[:2]
            cv2.putText(
                frame,
                label,
                (w_img - fw - 16, h_img - 16),
                FONT,
                0.5,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
