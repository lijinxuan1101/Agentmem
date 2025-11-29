from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, List, Any

from .config import OCRConfig
from .ocr import get_ocr_client
from .types import Anchor, VisualToken

_ENTITY_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9\-]+|\d{2,})\b")


@dataclass
class AnchorGenerator:
    """Creates OCR-inspired anchors from visual tokens."""

    timestamp_scale: float = 60.0  # seconds
    ocr_config: OCRConfig = field(default_factory=OCRConfig)
    _client: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._client = get_ocr_client(self.ocr_config)

    def generate(self, tokens: Iterable[VisualToken], *, base_time: float | None = None) -> List[Anchor]:
        try:
            return self._client.extract(tokens)
        except Exception:
            # Fallback to regex extraction if OCR backend fails
            return self._fallback(tokens, base_time=base_time)

    def _fallback(self, tokens: Iterable[VisualToken], *, base_time: float | None) -> List[Anchor]:
        base_time = base_time or time.time()
        anchors: List[Anchor] = []
        for idx, token in enumerate(tokens):
            matches = _ENTITY_PATTERN.findall(token.text_span)
            if not matches:
                continue
            timestamp = base_time + idx * self.timestamp_scale
            anchors.append(
                Anchor(
                    anchor_id=f"an-{uuid.uuid4().hex[:8]}",
                    token_id=token.token_id,
                    text=" ".join(matches[:6]),
                    entities=matches[:10],
                    timestamp=timestamp,
                    confidence=min(0.99, 0.5 + 0.5 * token.saliency),
                )
            )
        return anchors
