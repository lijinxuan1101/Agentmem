from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Iterable, List

from .config import VisualizerConfig
from .types import VisualToken
from .vision import PseudoViTEncoder, VISTLikeRenderer
from .vist_backend import get_vist_backend, VISTBackend


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _chunks(words: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(words), size):
        yield words[idx : idx + size]


@dataclass
class VisualSketchBuffer:
    """Transforms raw text into pseudo-visual tokens with simple heuristics."""

    config: VisualizerConfig = field(default_factory=VisualizerConfig)
    _renderer: VISTLikeRenderer | None = field(init=False, default=None)
    _encoder: PseudoViTEncoder | None = field(init=False, default=None)
    _vist_backend: VISTBackend | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.config.render_mode != "text":
            self._renderer = VISTLikeRenderer(self.config)
            self._encoder = PseudoViTEncoder(self.config)
            self._vist_backend = get_vist_backend(self.config)

    def render(self, text: str, canvas_id: str | None = None) -> List[VisualToken]:
        if self.config.render_mode == "vist" and self._vist_backend:
            return self._render_backend(text, canvas_id)
        if self.config.render_mode == "vist" and self._renderer and self._encoder:
            return self._render_vist(text, canvas_id)
        clean_text = _normalize_whitespace(text)
        words = clean_text.split()
        canvas_id = canvas_id or f"canvas-{uuid.uuid4().hex[:8]}"
        tokens: List[VisualToken] = []

        for idx, chunk in enumerate(_chunks(words, self.config.paragraph_tokens)):
            span = " ".join(chunk)
            unique_ratio = len(set(chunk)) / max(len(chunk), 1)
            capitals = sum(1 for w in chunk if w[:1].isupper())
            digits = sum(1 for w in chunk if w.isdigit())
            saliency = min(1.0, unique_ratio + 0.1 * capitals + 0.05 * digits)

            grid_w = grid_h = int(math.sqrt(max(1, math.ceil(len(words) / self.config.paragraph_tokens)))) + 1
            x = idx % grid_w
            y = idx // grid_w

            tokens.append(
                VisualToken(
                    token_id=f"vt-{uuid.uuid4().hex[:8]}",
                    canvas_id=canvas_id,
                    text_span=span,
                    saliency=saliency,
                    resolution=self.config.base_resolution,
                    layout_pos=(x, y % grid_h),
                    metadata={
                        "unique_ratio": unique_ratio,
                        "capital_density": capitals / max(len(chunk), 1),
                        "digit_density": digits / max(len(chunk), 1),
                        "chunk_index": float(idx),
                    },
                )
            )

        return tokens

    def _render_backend(self, text: str, canvas_id: str | None) -> List[VisualToken]:
        assert self._vist_backend
        patches = self._vist_backend.render(text, canvas_id)
        tokens: List[VisualToken] = []

        for patch in patches:
            embedding = patch.get("embedding", [])
            tokens.append(
                VisualToken(
                    token_id=f"vt-{uuid.uuid4().hex[:8]}",
                    canvas_id=canvas_id or f"canvas-{uuid.uuid4().hex[:8]}",
                    text_span=patch.get("text", ""),
                    saliency=patch.get("saliency", 0.5),
                    resolution=patch.get("resolution", self.config.base_resolution),
                    layout_pos=patch.get("layout", (0, 0)),
                    metadata={
                        "embedding": embedding,
                        "backend": self.config.render_backend,
                    },
                )
            )

        return tokens

    def _render_vist(self, text: str, canvas_id: str | None) -> List[VisualToken]:
        assert self._renderer and self._encoder
        clean_text = _normalize_whitespace(text)
        canvas = self._renderer.render_canvas(clean_text, canvas_id)
        patches = self._encoder.encode(canvas)
        tokens: List[VisualToken] = []

        for patch in patches:
            resolution = (
                int(self.config.base_resolution[0] * (patch["resolution"][0] / self.config.patch_char_size)),
                int(self.config.base_resolution[1] * (patch["resolution"][1] / self.config.patch_char_size)),
            )
            tokens.append(
                VisualToken(
                    token_id=f"vt-{uuid.uuid4().hex[:8]}",
                    canvas_id=canvas.canvas_id,
                    text_span=patch["text"],
                    saliency=patch["saliency"],
                    resolution=resolution,
                    layout_pos=patch["layout"],
                    metadata={
                        "unique_ratio": patch["unique_ratio"],
                        "capital_density": patch["upper_ratio"],
                        "digit_density": patch["digit_ratio"],
                        "chunk_index": float(patch["level"]),
                        "char_density": patch["char_density"],
                        "alnum_ratio": patch["alnum_ratio"],
                        "resolution_chars": float(patch["resolution"][0]),
                    },
                )
            )

        return tokens
