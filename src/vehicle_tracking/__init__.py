"""Vehicle detection, tracking and per-lane counting (YOLOv3 + Deep SORT)."""

from vehicle_tracking.config import Config, LaneConfig, load_config
from vehicle_tracking.counter import CountEvent, LaneCounter
from vehicle_tracking.lanes import Lane
from vehicle_tracking.types import Detection, TrackedVehicle

__version__ = "1.0.0"

__all__ = [
    "Config",
    "CountEvent",
    "Detection",
    "Lane",
    "LaneConfig",
    "LaneCounter",
    "TrackedVehicle",
    "load_config",
]
