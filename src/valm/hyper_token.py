from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from .config import HyperTokenizerConfig
from .entropy import SemanticEntropyEstimator
from .types import Anchor, HyperToken, VisualToken


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


@dataclass
class HyperTokenizer:
    """Compresses visual tokens + anchors into latent HyperTokens."""

    entropy_estimator: SemanticEntropyEstimator
    config: HyperTokenizerConfig = field(default_factory=HyperTokenizerConfig)

    def __post_init__(self) -> None:
        random.seed(self.config.random_seed)

    def encode(self, token: VisualToken, anchor: Anchor | None = None) -> HyperToken | None:
        entropy = self.entropy_estimator.estimate(token)
        importance = self._importance_score(token, anchor, entropy)
        if importance < self.config.drop_threshold:
            return None

        bits = self.entropy_estimator.bit_allocation(entropy)
        if importance >= self.config.high_def_threshold:
            bits = min(
                self.entropy_estimator.config.max_bit,
                bits + self.config.bit_boost,
            )
        feature_vec = self._build_feature_vector(token, anchor, entropy)
        quantized = tuple(self._quantize_vector(feature_vec, bits))
        metadata = {
            "entropy": entropy,
            "bits": float(bits),
            "saliency": token.saliency,
            "anchor_confidence": anchor.confidence if anchor else 0.0,
            "timestamp": anchor.timestamp if anchor else 0.0,
            "importance": importance,
        }

        return HyperToken(
            token_id=token.token_id,
            quantized_vector=quantized,
            bit_allocation=[bits] * len(quantized),
            metadata=metadata,
        )

    def _importance_score(self, token: VisualToken, anchor: Anchor | None, entropy: float) -> float:
        meta = token.metadata
        unique_ratio = meta.get("unique_ratio", 0.0)
        capital_density = meta.get("capital_density", 0.0)
        digit_density = meta.get("digit_density", 0.0)
        char_density = meta.get("char_density", 0.0)
        entropy_span = max(1e-3, self.entropy_estimator.config.entropy_ceiling)
        entropy_norm = min(1.0, entropy / entropy_span)
        anchor_conf = anchor.confidence if anchor else 0.0

        score = (
            0.3 * token.saliency
            + 0.2 * unique_ratio
            + 0.1 * capital_density
            + 0.15 * digit_density
            + 0.15 * entropy_norm
            + 0.05 * char_density
            + 0.05 * anchor_conf
        )
        return max(0.0, min(1.0, score))

    def _build_feature_vector(
        self, token: VisualToken, anchor: Anchor | None, entropy: float
    ) -> Sequence[float]:
        meta = token.metadata
        unique_ratio = meta.get("unique_ratio", 0.0)
        capital_density = meta.get("capital_density", 0.0)
        digit_density = meta.get("digit_density", 0.0)
        chunk_index = meta.get("chunk_index", 0.0)
        chunk_norm = math.tanh(chunk_index / 10.0)

        features = [
            token.saliency,
            unique_ratio,
            capital_density,
            digit_density,
            _sigmoid(entropy),
            chunk_norm,
        ]

        if anchor:
            entity_density = _safe_div(len(anchor.entities), 10.0)
            features.extend(
                [
                    anchor.confidence,
                    _sigmoid(entity_density),
                    _sigmoid(anchor.timestamp % (24 * 3600) / (24 * 3600)),
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0])

        return features

    def _quantize_vector(self, vector: Sequence[float], bits: int) -> Sequence[int]:
        scale = max(1, (1 << bits) - 1)
        quantized = []
        for value in vector:
            normalized = max(0.0, min(1.0, value))
            # Introduce mild stochastic rounding to imitate residual coding.
            noise = random.random() / scale
            q_value = int(round((normalized + noise) * scale))
            quantized.append(min(scale, max(0, q_value)))
        return quantized
