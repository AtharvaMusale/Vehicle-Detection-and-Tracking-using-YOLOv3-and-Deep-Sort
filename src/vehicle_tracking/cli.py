"""Command line interface: ``vehicle-tracking run|preview-lanes|download-assets``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from vehicle_tracking.config import Config, load_config

logger = logging.getLogger("vehicle_tracking")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vehicle-tracking", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="detect, track and count vehicles in a video")
    run.add_argument("--video", type=Path, required=True)
    run.add_argument("--output", type=Path, default=Path("outputs/demo.mp4"))
    run.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    run.add_argument("--max-frames", type=int, help="stop after N frames (quick tests)")
    run.add_argument("--start", type=float, help="start offset in seconds")

    preview = sub.add_parser(
        "preview-lanes", help="save one frame with lanes drawn, to tune the config"
    )
    preview.add_argument("--video", type=Path, required=True)
    preview.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    preview.add_argument("--output", type=Path, default=Path("outputs/lanes_preview.jpg"))
    preview.add_argument("--at", type=float, default=0.0, help="timestamp in seconds")

    dl = sub.add_parser("download-assets", help="download YOLOv3 cfg and weights")
    dl.add_argument("--models-dir", type=Path, default=Path("models"))
    return parser


def _override(config: Config, args: argparse.Namespace) -> Config:
    from dataclasses import replace

    video = config.video
    if args.max_frames is not None:
        video = replace(video, max_frames=args.max_frames)
    if args.start is not None:
        video = replace(video, start_seconds=args.start)
    return replace(config, video=video)


def _cmd_run(args: argparse.Namespace) -> int:
    from vehicle_tracking.detector import YoloV3Detector
    from vehicle_tracking.pipeline import run_pipeline

    config = _override(load_config(args.config), args)
    detector = YoloV3Detector(config.detector)

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            pct = f" ({100 * done / total:.0f}%)" if total else ""
            print(f"\rframe {done}/{total}{pct}", end="", file=sys.stderr, flush=True)

    summary = run_pipeline(config, detector, args.video, args.output, progress)
    print(file=sys.stderr)
    print(json.dumps(summary, indent=2))
    logger.info("Wrote %s", args.output)
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    from vehicle_tracking.counter import LaneCounter
    from vehicle_tracking.video_io import VideoReader
    from vehicle_tracking.visualize import Renderer

    config = load_config(args.config)
    reader = VideoReader(args.video, config.video.process_width, args.at)
    frame = next(iter(reader), None)
    reader.release()
    if frame is None:
        logger.error("Could not read a frame at %.1fs", args.at)
        return 1
    lanes = config.build_lanes(reader.width, reader.height)
    out = Renderer(config.render, lanes).draw(frame, [], LaneCounter(lanes), [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), out)
    print(f"Saved {args.output}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from vehicle_tracking.assets import download_assets

    for path in download_assets(args.models_dir):
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    handlers = {"run": _cmd_run, "preview-lanes": _cmd_preview, "download-assets": _cmd_download}
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
