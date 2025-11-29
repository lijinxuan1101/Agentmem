from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class VisualToken:
    """Lightweight representation of a pseudo-visual patch derived from text."""

    token_id: str
    canvas_id: str
    text_span: str
    saliency: float
    resolution: tuple[int, int]
    layout_pos: tuple[int, int]
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Anchor:
    """OCR-inspired anchor that links a visual token back to symbolic text."""

    anchor_id: str
    token_id: str
    text: str
    entities: List[str]
    timestamp: float
    confidence: float


@dataclass(frozen=True)
class HyperToken:
    """Quantized latent memory carrier that stores residual code indices."""

    token_id: str
    quantized_vector: Sequence[int]
    bit_allocation: Sequence[int]
    metadata: Dict[str, float]


@dataclass(frozen=True)
class QueryContext:
    """Metadata describing a retrieval request."""

    text: str
    topic: str | None = None
    time_hint: float | None = None
    time_window: float | None = None
    metadata_filters: Dict[str, tuple[float | None, float | None]] | None = None

