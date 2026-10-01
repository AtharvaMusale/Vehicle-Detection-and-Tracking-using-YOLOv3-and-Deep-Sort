"""Video reading and H.264 writing (ffmpeg when available, OpenCV otherwise)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from vehicle_tracking.types import Array

logger = logging.getLogger(__name__)


class VideoReader:
    def __init__(self, path: Path, process_width: int, start_seconds: float = 0.0) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        src_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scale = min(1.0, process_width / src_w)
        # even dimensions are required by yuv420p
        self.width = int(round(src_w * scale / 2) * 2)
        self.height = int(round(src_h * scale / 2) * 2)
        if start_seconds > 0:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

    def __iter__(self) -> Iterator[Array]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            yield frame

    def release(self) -> None:
        self._cap.release()


class VideoWriter:
    """Pipes raw frames to ffmpeg (libx264, yuv420p) so output plays in browsers."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ffmpeg: subprocess.Popen[bytes] | None = None
        self._cv: cv2.VideoWriter | None = None
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                "-r", f"{fps:.3f}", "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
            ]  # fmt: skip
            self._ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            logger.warning("ffmpeg not found; falling back to OpenCV mp4v encoding")
            self._cv = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter.fourcc(*"mp4v"),
                fps,
                (width, height),
            )

    def write(self, frame: Array) -> None:
        if self._ffmpeg is not None and self._ffmpeg.stdin is not None:
            self._ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
        elif self._cv is not None:
            self._cv.write(frame)

    def close(self) -> None:
        if self._ffmpeg is not None and self._ffmpeg.stdin is not None:
            self._ffmpeg.stdin.close()
            self._ffmpeg.wait()
        if self._cv is not None:
            self._cv.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
