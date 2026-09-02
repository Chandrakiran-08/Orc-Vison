"""YAML config loader for ``orcvision run --config FILE``.

The config mirrors the CLI options so a run can be fully described in one
file. See the ``examples/`` directory for complete configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SensorConfig:
    type: str = "rgb"  # rgb | realsense
    source: Any = 0  # int, path, or rtsp:// URL
    width: int | None = None
    height: int | None = None


@dataclass
class ModelConfig:
    name: str = "rtdetr-l"
    weights: str | None = None
    confidence: float = 0.35


@dataclass
class TrackerConfig:
    enabled: bool = False


@dataclass
class DepthConfig:
    enabled: bool = False


@dataclass
class DecisionConfig:
    rules: list[dict[str, Any]] = field(default_factory=list)
    event_rules: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SinkConfig:
    type: str = "stdout"  # stdout | mqtt | file
    # mqtt
    host: str = "localhost"
    port: int = 1883
    topic: str = "orcvision/events"
    # file
    path: str = "orcvision_events.jsonl"


@dataclass
class RunConfig:
    sensor: SensorConfig = field(default_factory=SensorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    sink: SinkConfig = field(default_factory=SinkConfig)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")

    return RunConfig(
        sensor=SensorConfig(**(data.get("sensor") or {})),
        model=ModelConfig(**(data.get("model") or {})),
        tracker=TrackerConfig(**(data.get("tracker") or {})),
        depth=DepthConfig(**(data.get("depth") or {})),
        decision=DecisionConfig(**(data.get("decision") or {})),
        sink=SinkConfig(**(data.get("sink") or {})),
    )
