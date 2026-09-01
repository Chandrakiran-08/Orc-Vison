"""Stdout sink — prints each PerceptionEvent as a single JSON line."""

from __future__ import annotations

import sys

from orcvision.events import PerceptionEvent


class StdoutSink:
    """Write newline-delimited JSON events to stdout."""

    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def emit(self, event: PerceptionEvent) -> None:
        self._stream.write(event.to_json() + "\n")
        self._stream.flush()

    def close(self) -> None:  # nothing to release for stdout
        pass
