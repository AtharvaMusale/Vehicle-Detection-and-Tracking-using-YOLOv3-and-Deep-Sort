"""Download the pretrained model files (not stored in git)."""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

ASSETS = {
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
}
EXPECTED_WEIGHTS_BYTES = 248_007_048


def _fetch(url: str, destination: Path) -> None:
    # Some hosts reject urllib's default User-Agent with HTTP 403.
    request = urllib.request.Request(url, headers={"User-Agent": "vehicle-tracking/1.0"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(response, out, length=1 << 20)


def download_assets(models_dir: Path) -> list[Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, url in ASSETS.items():
        target = models_dir / name
        paths.append(target)
        if target.is_file() and target.stat().st_size > 0:
            logger.info("%s already present, skipping", name)
            continue
        logger.info("Downloading %s from %s", name, url)
        partial = target.with_suffix(target.suffix + ".part")
        _fetch(url, partial)
        partial.replace(target)
    weights = models_dir / "yolov3.weights"
    if weights.stat().st_size != EXPECTED_WEIGHTS_BYTES:
        logger.warning(
            "yolov3.weights is %d bytes (expected %d); the download may be incomplete",
            weights.stat().st_size,
            EXPECTED_WEIGHTS_BYTES,
        )
    return paths
