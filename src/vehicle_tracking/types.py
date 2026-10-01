"""Plain data containers shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detector output in pixel coordinates (left, top, width, height)."""

    ltwh: tuple[float, float, float, float]
    score: float
    label: str


@dataclass(frozen=True)
class TrackedVehicle:
    """A confirmed track in the current frame."""

    track_id: int
    ltrb: tuple[float, float, float, float]
    label: str
    score: float

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the vehicle touches the road."""
        left, _, right, bottom = self.ltrb
        return ((left + right) / 2.0, bottom)

    def anchor_array(self) -> np.ndarray:
        return np.asarray(self.anchor, dtype=np.float32)
