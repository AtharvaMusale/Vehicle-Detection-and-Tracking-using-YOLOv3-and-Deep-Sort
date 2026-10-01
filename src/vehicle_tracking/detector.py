"""YOLOv3 vehicle detector running on OpenCV's DNN module (same Darknet weights)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

from vehicle_tracking.config import DetectorConfig
from vehicle_tracking.types import Array, Detection

logger = logging.getLogger(__name__)


def load_class_names(path: Path | None = None) -> list[str]:
    if path is not None:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    text = resources.files("vehicle_tracking").joinpath("data/coco.names").read_text()
    return [line.strip() for line in text.splitlines() if line.strip()]


class YoloV3Detector:
    def __init__(self, config: DetectorConfig) -> None:
        for path in (config.cfg_path, config.weights_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing model file: {path}. Run `vehicle-tracking download-assets`."
                )
        self.config = config
        self.names = load_class_names(config.names_path)
        unknown = set(config.classes) - set(self.names)
        if unknown:
            raise ValueError(f"Unknown class names in config: {sorted(unknown)}")
        self._wanted = {self.names.index(c) for c in config.classes}
        self._net = cv2.dnn.readNetFromDarknet(str(config.cfg_path), str(config.weights_path))
        self._output_layers = list(self._net.getUnconnectedOutLayersNames())
        logger.info("Loaded YOLOv3 (%d output layers)", len(self._output_layers))

    def detect(self, frame: Array) -> list[Detection]:
        height, width = frame.shape[:2]
        size = self.config.input_size
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (size, size), swapRB=True, crop=False)
        self._net.setInput(blob)
        outputs = self._net.forward(self._output_layers)
        return self._postprocess([np.asarray(o) for o in outputs], width, height)

    def _postprocess(self, outputs: Sequence[Array], width: int, height: int) -> list[Detection]:
        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for output in outputs:
            for row in output:
                class_scores = row[5:]
                class_id = int(np.argmax(class_scores))
                if class_id not in self._wanted:
                    continue
                score = float(row[4] * class_scores[class_id])
                if score < self.config.score_threshold:
                    continue
                cx, cy, w, h = row[0] * width, row[1] * height, row[2] * width, row[3] * height
                boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
                scores.append(score)
                class_ids.append(class_id)
        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxesBatched(
            boxes, scores, class_ids, self.config.score_threshold, self.config.nms_threshold
        )
        return [
            Detection(
                ltwh=tuple(float(v) for v in boxes[i]),  # type: ignore[arg-type]
                score=scores[i],
                label=self.names[class_ids[i]],
            )
            for i in np.asarray(keep).flatten()
        ]
