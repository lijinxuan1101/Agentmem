from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

from .config import RetrievalConfig
from .store import VALMStore
from .types import Anchor, HyperToken, QueryContext, VisualToken
from .debug_utils import log_debug_json

_WORD_RE = re.compile(r"[A-Za-z0-9\-]+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _text_overlap_score(query: str, source: str) -> float:
    q_words = set(_tokenize(query))
    if not q_words:
        return 0.0
    s_words = set(_tokenize(source))
    if not s_words:
        return 0.0
    return len(q_words & s_words) / len(q_words)


def _query_vector(text: str, length: int) -> List[int]:
    tokens = _tokenize(text)
    if not tokens:
        return [0] * length
    base = sum(ord(ch) for ch in text) % 997
    vec = []
    for idx in range(length):
        token = tokens[idx % len(tokens)]
        value = (base + ord(token[0]) * (idx + 1)) % 256
        vec.append(value)
    return vec


@dataclass
class VALMRetrievalEngine:
    """Hybrid retrieval over textual anchors and residual latent codes."""

    store: VALMStore
    config: RetrievalConfig = field(default_factory=RetrievalConfig)

    def retrieve(self, query: QueryContext) -> List[Tuple[HyperToken, float]]:
        candidates = list(self._candidate_hyper_tokens(query))
        if not candidates:
            return []

        ranked: List[Tuple[HyperToken, float]] = []
        query_heads = self._build_query_heads(query.text, len(candidates[0].quantized_vector))

        for hyper in candidates:
            if query.metadata_filters and not self._passes_metadata_filters(hyper, query.metadata_filters):
                continue

            anchor = self._anchor_for_token(hyper.token_id)
            token = self.store.tokens.get(hyper.token_id)

            if query.time_hint and query.time_window and anchor:
                if abs(anchor.timestamp - query.time_hint) > query.time_window:
                    continue

            text_source = anchor.text if anchor else (token.text_span if token else hyper.metadata.get("event_text", ""))
            text_score = _text_overlap_score(query.text, text_source)

            residual_score = self._multi_head_residual(query_heads, hyper.quantized_vector)
            time_bonus = self._time_alignment(query.time_hint, anchor, hyper)
            topic_bonus = self._topic_bonus(query.topic, hyper.token_id)

            score = (
                self.config.metadata_weight * (text_score + time_bonus)
                + self.config.residual_weight * residual_score
                + topic_bonus
            )
            ranked.append((hyper, score))

        ranked.sort(key=lambda item: item[1], reverse=True)

        top_k = ranked[: self.config.coarse_top_k]

        # Debug：记录本次检索的 Top-K 结果
        debug_entries = []
        for hyper, s in top_k[:5]:
            anchor = self._anchor_for_token(hyper.token_id)
            token = self.store.tokens.get(hyper.token_id)
            text_source = anchor.text if anchor else (token.text_span if token else hyper.metadata.get("event_text", ""))
            debug_entries.append(
                {
                    "token_id": hyper.token_id,
                    "score": s,
                    "text": text_source,
                    "timestamp": hyper.metadata.get("timestamp"),
                }
            )
        log_debug_json(
            "locomo_retrieval.json",
            {
                "query_text": query.text,
                "topic": query.topic,
                "time_hint": query.time_hint,
                "time_window": query.time_window,
                "top_k": debug_entries,
            },
        )

        return top_k

    def _candidate_hyper_tokens(self, query: QueryContext) -> Iterable[HyperToken]:
        yielded = False
        if query.topic:
            for token in self.store.tokens_for_topic(query.topic):
                if token.token_id in self.store.hyper_tokens:
                    hyper = self.store.hyper_tokens[token.token_id]
                    if hyper.metadata.get("importance", 0.0) >= self.config.importance_threshold:
                        yielded = True
                        yield hyper
            for hyper in self.store.event_tokens(query.topic):
                yield hyper
        else:
            for hyper in self.store.hyper_tokens.values():
                if hyper.metadata.get("importance", 0.0) >= self.config.importance_threshold:
                    yielded = True
                    yield hyper
        if not yielded:
            for hyper in self.store.event_tokens(query.topic):
                yield hyper

    def _anchor_for_token(self, token_id: str) -> Anchor | None:
        for anchor in self.store.anchors.values():
            if anchor.token_id == token_id:
                return anchor
        return None

    def _multi_head_residual(self, query_heads: List[List[int]], hyper_vec: Iterable[int]) -> float:
        scores = []
        for head in query_heads:
            diffs = [abs(q - h) for q, h in zip(head, hyper_vec)]
            if not diffs:
                continue
            norm = sum(diffs) / len(diffs)
            scores.append(1.0 / (1.0 + norm))
        return sum(scores) / len(scores) if scores else 0.0

    def _build_query_heads(self, text: str, length: int) -> List[List[int]]:
        heads: List[List[int]] = []
        for head_idx in range(max(1, self.config.head_count)):
            offset_text = f"{text}#{head_idx}"
            heads.append(_query_vector(offset_text, length))
        return heads

    def _time_alignment(self, time_hint: float | None, anchor: Anchor | None, hyper: HyperToken) -> float:
        if time_hint is None:
            return 0.0
        timestamp = None
        if anchor:
            timestamp = anchor.timestamp
        elif hyper.metadata.get("timestamp"):
            timestamp = hyper.metadata.get("timestamp")
        if timestamp is None:
            return 0.0
        delta = abs(time_hint - timestamp)
        return math.exp(-delta / max(1.0, self.config.time_decay))

    def _topic_bonus(self, topic: str | None, token_id: str) -> float:
        if topic and self.store.token_in_topic(token_id, topic):
            return self.config.topic_boost
        return 0.0

    @staticmethod
    def _passes_metadata_filters(
        hyper: HyperToken, filters: dict[str, tuple[float | None, float | None]]
    ) -> bool:
        for key, (lower, upper) in filters.items():
            value = hyper.metadata.get(key)
            if value is None:
                return False
            if lower is not None and value < lower:
                return False
            if upper is not None and value > upper:
                return False
        return True
