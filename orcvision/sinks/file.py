"""File sink — appends each PerceptionEvent as a JSON line (JSONL)."""

from __future__ import annotations

from pathlib import Path

from orcvision.events import PerceptionEvent


class FileSink:
    """Write newline-delimited JSON events to a ``.jsonl`` file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, event: PerceptionEvent) -> None:
        self._fh.write(event.to_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
