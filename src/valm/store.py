from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .types import Anchor, HyperToken, VisualToken
from .debug_utils import log_debug_json

_KV_PATTERN = re.compile(r"(key_\d+_\d+)\s+is\s+([A-Za-z0-9_\-]+)")
_ALERT_PATTERN = re.compile(r"(\d{10})\s+ALERT")
_DATE_FORMATS = ["%d %B, %Y", "%B %d, %Y", "%Y-%m-%d"]


@dataclass
class VALMStore:
    """In-memory prototype store for visual tokens, anchors, and HyperTokens."""

    tokens: Dict[str, VisualToken] = field(default_factory=dict)
    anchors: Dict[str, Anchor] = field(default_factory=dict)
    hyper_tokens: Dict[str, HyperToken] = field(default_factory=dict)
    topic_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    token_topics: Dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    kv_index: Dict[str, Dict[str, str]] = field(default_factory=lambda: defaultdict(dict))
    alert_index: Dict[str, List[dict]] = field(default_factory=lambda: defaultdict(list))
    event_index: Dict[str, List[dict]] = field(default_factory=lambda: defaultdict(list))
    event_hyper_tokens: Dict[str, List[HyperToken]] = field(default_factory=lambda: defaultdict(list))
    hyper_token_dim: int = 0

    def add_tokens(self, tokens: Iterable[VisualToken], topic: str | None = None) -> None:
        for token in tokens:
            self.tokens[token.token_id] = token
            if topic:
                self.topic_index[topic].append(token.token_id)
                self.token_topics[token.token_id].add(topic)
                self._update_structured_indexes(token.text_span, topic, token.token_id)

    def add_anchors(self, anchors: Iterable[Anchor]) -> None:
        for anchor in anchors:
            self.anchors[anchor.anchor_id] = anchor
            if anchor.text and anchor.token_id:
                for topic in self.token_topics.get(anchor.token_id, []):
                    self._update_structured_indexes(anchor.text, topic, anchor.token_id)

    def add_hyper_tokens(self, hyper_tokens: Iterable[HyperToken]) -> None:
        for hyper in hyper_tokens:
            self.hyper_tokens[hyper.token_id] = hyper
            if not self.hyper_token_dim and hyper.quantized_vector:
                self.hyper_token_dim = len(hyper.quantized_vector)

    def index_raw_text(self, text: str, topic: str | None) -> None:
        if not topic or not text:
            return
        self._update_structured_indexes(text, topic, None)

    def lookup_key_value(self, topic: str | None, key: str) -> Optional[str]:
        if topic is None:
            return None
        return self.kv_index.get(topic, {}).get(key)

    def latest_alert(self, topic: str | None) -> Optional[dict]:
        if topic is None:
            return None
        alerts = self.alert_index.get(topic, [])
        if not alerts:
            return None
        return max(alerts, key=lambda item: item["timestamp"])

    def anchors_for_token(self, token_id: str) -> List[Anchor]:
        return [a for a in self.anchors.values() if a.token_id == token_id]

    def tokens_for_topic(self, topic: str) -> List[VisualToken]:
        return [self.tokens[token_id] for token_id in self.topic_index.get(topic, []) if token_id in self.tokens]

    def token_in_topic(self, token_id: str, topic: str | None) -> bool:
        if topic is None:
            return False
        return topic in self.token_topics.get(token_id, set())

    def _update_structured_indexes(self, text: str, topic: str, token_id: str | None) -> None:
        if not text:
            return
        for key, value in _KV_PATTERN.findall(text):
            self.kv_index[topic][key] = value
        for match in _ALERT_PATTERN.findall(text):
            self.alert_index[topic].append({
                "timestamp": float(match),
                "token_id": token_id,
                "text": text,
            })

    def add_event_summary(self, topic: str | None, event_summary: dict | None) -> None:
        if not topic or not event_summary or topic in self.event_index:
            return
        events: List[dict] = []
        for details in event_summary.values():
            if not isinstance(details, dict):
                continue
            date = details.get("date")
            timestamp = _parse_date_to_timestamp(date)
            for speaker, items in details.items():
                if speaker == "date":
                    continue
                sentences = items if isinstance(items, list) else [items]
                for sentence in sentences:
                    if isinstance(sentence, str) and sentence.strip():
                        events.append(
                            {
                                "text": sentence.strip(),
                                "date": date,
                                "timestamp": timestamp,
                            }
                        )
        if events:
            self.event_index[topic].extend(events)
            self._materialize_event_tokens(topic, events)
            # Debug：记录每个 topic 的结构化事件
            log_debug_json(
                "locomo_events.json",
                {
                    "topic": topic,
                    "events": events,
                },
            )

    def events_for_topic(self, topic: str | None) -> List[dict]:
        if topic is None:
            return []
        return self.event_index.get(topic, [])

    def match_event(self, topic: str | None, question: str) -> Tuple[Optional[dict], float]:
        events = self.events_for_topic(topic)
        if not events:
            return None, 0.0
        question_text = str(question or "")
        q_tokens = set(question_text.lower().split())
        best_score = 0.0
        best_event: Optional[dict] = None
        for event in events:
            text = event.get("text")
            if not text:
                continue
            e_tokens = set(text.lower().split())
            if not q_tokens or not e_tokens:
                continue
            overlap = len(q_tokens & e_tokens) / len(q_tokens)
            if overlap > best_score:
                best_score = overlap
                best_event = event
        # Debug：记录事件匹配过程
        log_debug_json(
            "locomo_match_event.json",
            {
                "topic": topic,
                "question": question_text,
                "best_event": best_event,
                "best_score": best_score,
                "total_events": len(events),
            },
        )
        return best_event, best_score

    def event_tokens(self, topic: str | None) -> List[HyperToken]:
        if topic is None:
            return []
        return self.event_hyper_tokens.get(topic, [])

    def _materialize_event_tokens(self, topic: str, events: List[dict]) -> None:
        dim = self.hyper_token_dim or 8
        vector = tuple(0 for _ in range(dim))
        bits = [0] * dim
        for idx, event in enumerate(events):
            token_id = f"event-{topic}-{len(self.event_hyper_tokens[topic]) + idx}"
            importance = 1.0
            metadata = {
                "importance": importance,
                "entropy": 1.0,
                "saliency": 1.0,
                "timestamp": event.get("timestamp") or 0.0,
                "event_text": event.get("text", ""),
                "event_date": event.get("date", ""),
            }
            event_token = HyperToken(token_id=token_id, quantized_vector=vector, bit_allocation=bits, metadata=metadata)
            self.event_hyper_tokens[topic].append(event_token)



def _parse_date_to_timestamp(date_str: Optional[str]) -> Optional[float]:
    if not date_str:
        return None
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.timestamp()
        except ValueError:
            continue
    return None
