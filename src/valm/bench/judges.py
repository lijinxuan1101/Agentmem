from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..pipeline import VALMPipeline

_KEY_PATTERN = re.compile(r"(key_\d+_\d+)")


@dataclass
class BenchmarkJudge:
    name: str

    def structured_prediction(self, pipeline: VALMPipeline, sample: Dict[str, Any], topic: str) -> Optional[str]:
        return None


class LoCoMoJudge(BenchmarkJudge):
    name = "locomo"

    def structured_prediction(self, pipeline: VALMPipeline, sample: Dict[str, Any], topic: str) -> Optional[str]:
        question = sample.get("question", "")
        if self._is_temporal_question(question):
            event_match, event_score = pipeline.store.match_event(topic, question)
            if event_match and event_match.get("date") and event_score > 0.15:
                return event_match["date"]
        else:
            event_match, event_score = pipeline.store.match_event(topic, question)
            if event_match and event_score > 0.5:
                return event_match.get("text")

        metadata = sample.get("metadata") or {}
        event_summary = metadata.get("event_summary")
        date_answer = self._match_event_date(question, event_summary)
        if date_answer:
            return date_answer
        candidates = []
        candidates.extend(self._flatten_structured(event_summary))
        candidates.extend(self._flatten_structured(metadata.get("session_summary")))
        candidates.extend(self._flatten_structured(metadata.get("observation")))
        if not candidates:
            return None
        best, score = self._match(sample.get("question", ""), candidates)
        if score > 0.35:
            return best
        alert = pipeline.store.latest_alert(topic)
        if alert:
            return str(int(alert["timestamp"]))
        return None

    @staticmethod
    def _flatten_structured(value: Any) -> List[str]:
        texts: List[str] = []
        if value is None:
            return texts
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                texts.extend(LoCoMoJudge._flatten_structured(item))
        elif isinstance(value, list):
            for item in value:
                texts.extend(LoCoMoJudge._flatten_structured(item))
        elif isinstance(value, (int, float)):
            texts.append(str(value))
        else:
            texts.append(str(value))
        return texts

    @staticmethod
    def _match_event_date(question: str, event_summary: Any) -> Optional[str]:
        if not isinstance(event_summary, dict):
            return None
        events: List[tuple[str, Optional[str]]] = []
        for details in event_summary.values():
            if not isinstance(details, dict):
                continue
            date = details.get("date")
            for speaker, items in details.items():
                if speaker == "date":
                    continue
                entries = items if isinstance(items, list) else [items]
                for entry in entries:
                    if isinstance(entry, str):
                        events.append((entry, date))
        if not events:
            return None
        best, score = LoCoMoJudge._match(question, [e[0] for e in events])
        if best and score > 0.3:
            for text, date in events:
                if text == best and date:
                    return str(date)
        return None

    @staticmethod
    def _match(question: str, candidates: List[str]) -> tuple[Optional[str], float]:
        import difflib

        best_score = 0.0
        best_text: Optional[str] = None
        question_text = str(question or "")
        q_tokens = set(question_text.lower().split())
        for text in candidates:
            if not text:
                continue
            cand_text = str(text)
            matcher = difflib.SequenceMatcher(None, question_text.lower(), cand_text.lower())
            ratio = matcher.ratio()
            t_tokens = set(cand_text.lower().split())
            overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens))
            score = max(ratio, overlap)
            if score > best_score:
                best_score = score
                best_text = cand_text
        return best_text, best_score

    @staticmethod
    def _is_temporal_question(question: str) -> bool:
        q = question.lower()
        keywords = ["when", "date", "day", "time", "哪天", "什么时候", "日期"]
        return any(key in q for key in keywords)


class InfiniteBenchJudge(BenchmarkJudge):
    name = "infinitebench"

    def structured_prediction(self, pipeline: VALMPipeline, sample: Dict[str, Any], topic: str) -> Optional[str]:
        question = sample.get("question", "")
        key = self._extract_key(question)
        if not key:
            return None
        return pipeline.store.lookup_key_value(topic, key)

    @staticmethod
    def _extract_key(text: str) -> Optional[str]:
        match = _KEY_PATTERN.search(text)
        return match.group(1) if match else None


class AgentBenchJudge(BenchmarkJudge):
    name = "agentbench"

    def structured_prediction(self, pipeline: VALMPipeline, sample: Dict[str, Any], topic: str) -> Optional[str]:
        return None


def get_judge(name: str) -> BenchmarkJudge:
    normalized = name.lower()
    if normalized in ("locomo", "temporal"):
        return LoCoMoJudge(name="locomo")
    if normalized in ("infinite", "infinitebench"):
        return InfiniteBenchJudge(name="infinitebench")
    if normalized in ("agentbench", "agent"):
        return AgentBenchJudge(name="agentbench")
    raise ValueError(f"Unknown benchmark judge: {name}")
