"""Monocular depth estimation via Depth Anything V2 (lazy, ``[depth]`` extra).

Enabled with ``--depth``. Produces a per-frame relative depth map; the
run loop samples the map at each detection's bbox centre to populate
``Detection.depth_m``. Note this is a *relative* monocular estimate, not
metric sensor depth — treat the value as approximate.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DepthEstimator:
    """Depth Anything V2 wrapper returning a normalized depth map."""

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf") -> None:
        try:
            import torch  # noqa: F401
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Depth estimation needs the optional extra: "
                "`pip install orcvision[depth]` (installs transformers + torch)."
            ) from exc

        import torch

        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline("depth-estimation", model=model_name, device=device)

    def estimate(self, frame: Any) -> np.ndarray:
        """Return a HxW float32 depth map aligned to ``frame``."""
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = self._pipe(Image.fromarray(rgb))
        depth = np.asarray(out["depth"], dtype=np.float32)
        if depth.shape[:2] != frame.shape[:2]:
            depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        return depth

    @staticmethod
    def sample_bbox(depth_map: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        """Sample the depth map at the centre of a bbox."""
        x1, y1, x2, y2 = bbox
        cx = min(max((x1 + x2) // 2, 0), depth_map.shape[1] - 1)
        cy = min(max((y1 + y2) // 2, 0), depth_map.shape[0] - 1)
        return float(depth_map[cy, cx])
