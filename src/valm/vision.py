from __future__ import annotations

import textwrap
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .config import VisualizerConfig


@dataclass(frozen=True)
class RenderedCanvas:
    canvas_id: str
    lines: Sequence[str]
    width: int
    height: int


class VISTLikeRenderer:
    """Renders text into a synthetic fixed-width canvas to emulate VIST layouts."""

    def __init__(self, config: VisualizerConfig):
        self.config = config

    def render_canvas(self, text: str, canvas_id: str | None = None) -> RenderedCanvas:
        canvas_id = canvas_id or f"canvas-{uuid.uuid4().hex[:8]}"
        truncated = text[: self.config.max_canvas_chars]
        wrapped = textwrap.wrap(truncated, self.config.wrap_width)
        if not wrapped:
            wrapped = [""]

        lines: List[str] = []
        for line in wrapped:
            padded = line.ljust(self.config.wrap_width)[: self.config.wrap_width]
            lines.append(padded)

        width = self.config.wrap_width
        height = len(lines)
        return RenderedCanvas(canvas_id=canvas_id, lines=lines, width=width, height=height)


class PseudoViTEncoder:
    """
    Encodes canvases into patch-level feature descriptors.

    Each patch approximates the behavior of a ViT token by aggregating character
    statistics inside a fixed-size window. Multiple resolutions are supported.
    """

    def __init__(self, config: VisualizerConfig):
        self.config = config

    def encode(self, canvas: RenderedCanvas) -> List[dict]:
        patches: List[dict] = []
        base_patch = max(4, self.config.patch_char_size)

        for level in range(max(1, self.config.multiscale_levels)):
            patch_size = base_patch * (2**level)
            for y in range(0, canvas.height, patch_size):
                for x in range(0, canvas.width, patch_size):
                    patch_lines = self._slice(canvas.lines, x, y, patch_size)
                    flat = "".join(patch_lines).strip()
                    if not flat:
                        continue

                    upper = sum(1 for ch in flat if ch.isupper())
                    digits = sum(1 for ch in flat if ch.isdigit())
                    alnum = sum(1 for ch in flat if ch.isalnum())
                    unique_chars = len(set(flat))
                    saliency = min(1.0, unique_chars / max(len(flat), 1) + 0.1 * upper + 0.05 * digits)
                    char_density = len(flat) / (patch_size * patch_size)

                    patches.append(
                        {
                            "text": flat[:256],
                            "saliency": saliency,
                            "layout": (x // patch_size, y // patch_size),
                            "resolution": (patch_size, patch_size),
                            "level": level,
                            "char_density": char_density,
                            "unique_ratio": unique_chars / max(len(flat), 1),
                            "upper_ratio": upper / max(len(flat), 1),
                            "digit_ratio": digits / max(len(flat), 1),
                            "alnum_ratio": alnum / max(len(flat), 1),
                        }
                    )

        return patches

    @staticmethod
    def _slice(lines: Sequence[str], x: int, y: int, size: int) -> Iterable[str]:
        for row in range(y, min(y + size, len(lines))):
            line = lines[row]
            yield line[x : min(x + size, len(line))]

