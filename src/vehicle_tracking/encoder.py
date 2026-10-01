"""Appearance features for Deep SORT's re-identification step."""

from __future__ import annotations

import cv2
import numpy as np

from vehicle_tracking.types import Detection


class ColorHistogramEncoder:
    """128-d HSV histogram descriptor.

    Light-weight stand-in for a learned appearance network: vehicles differ mostly
    by colour, and it needs no extra model file.
    """

    BINS = (8, 4, 4)  # 8*4*4 = 128 dimensions
    dim = 128

    def encode(self, frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        height, width = frame.shape[:2]
        features: list[np.ndarray] = []
        for det in detections:
            left, top, w, h = det.ltwh
            x1, y1 = max(int(left), 0), max(int(top), 0)
            x2, y2 = min(int(left + w), width), min(int(top + h), height)
            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros(self.dim, dtype=np.float32))
                continue
            hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, list(self.BINS), [0, 180, 0, 256, 0, 256])
            hist = hist.flatten().astype(np.float32)
            norm = float(np.linalg.norm(hist))
            features.append(hist / norm if norm > 0 else hist)
        return features
