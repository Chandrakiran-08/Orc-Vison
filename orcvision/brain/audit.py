"""Append-only decision audit trail.

When an autonomous machine does something unexpected, the first question is
always "why did it do that, and what did it know at the time?" An
explanation you can only see live is no use hours later, so every decision
can be written to a JSON Lines file as it is made.

Design constraints, from how these logs actually get used:

* **Append-only, one JSON object per line.** Survives a power cut mid-write
  with at most the last line lost, and streams into any log tooling.
* **Self-contained records.** Each line carries the action, the reasons with
  their weights, what was vetoed, the platform state and the situation key —
  enough to reconstruct the decision without the running process.
* **Bounded.** Rotates at a size cap so an unattended machine cannot fill
  its disk and take itself down.
* **Never fatal.** A logging failure must not stop the machine; write errors
  are counted and swallowed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from orcvision.brain.decision import Decision
    from orcvision.brain.state import WorldState


class AuditLog:
    """Records decisions to a rotating JSON Lines file."""

    def __init__(
        self,
        path: str | Path,
        max_bytes: int = 5_000_000,
        keep: int = 3,
        include_world: bool = True,
    ) -> None:
        self.path = Path(path).expanduser()
        self.max_bytes = max_bytes
        self.keep = keep
        self.include_world = include_world
        self.write_errors = 0
        self.records_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        decision: Decision,
        world: WorldState | None = None,
        timestamp: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Append one decision. Returns False if it could not be written."""
        entry: dict[str, Any] = {
            "t": timestamp if timestamp is not None else 0.0,
            "tick": world.tick if world is not None else None,
            "decision": decision.to_dict(),
        }
        if world is not None and self.include_world:
            platform = world.platform
            entry["goal"] = world.goal
            entry["objects"] = [
                {
                    "label": o.label,
                    "zone": o.zone,
                    "motion": o.motion,
                    "depth_m": o.depth_m,
                    "confidence": round(o.confidence, 3),
                }
                for o in world.visible()
            ]
            entry["platform"] = {
                "battery_pct": platform.battery_pct,
                "altitude_m": platform.altitude_m,
                "distance_from_home_m": platform.distance_from_home_m,
                "link_ok": platform.link_ok,
                "interlock_ok": platform.interlock_ok,
                "emergency": platform.emergency,
            }
        if extra:
            entry["extra"] = extra

        try:
            self._rotate_if_needed()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
            self.records_written += 1
            return True
        except OSError:
            # A full disk or read-only mount must never stop the machine.
            self.write_errors += 1
            return False

    def _rotate_if_needed(self) -> None:
        try:
            if not self.path.exists() or self.path.stat().st_size < self.max_bytes:
                return
        except OSError:  # pragma: no cover - stat race
            return
        # audit.jsonl -> audit.jsonl.1 -> .2 ... dropping the oldest.
        for index in range(self.keep - 1, 0, -1):
            older = self.path.with_suffix(self.path.suffix + f".{index}")
            newer = self.path.with_suffix(self.path.suffix + f".{index + 1}")
            if older.exists():
                os.replace(older, newer)
        os.replace(self.path, self.path.with_suffix(self.path.suffix + ".1"))

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Read records back, newest last. For post-incident review."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue  # tolerate a torn final line after a power cut
        return records[-limit:] if limit else records
