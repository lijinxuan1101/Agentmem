from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .base import BenchmarkAdapter, BenchmarkResult
from ..types import QueryContext


@dataclass
class LoCoMoAdapter(BenchmarkAdapter):
    """Prototype adapter that mimics a subset of LoCoMo behaviors."""

    name: str = "locomo"
    prepared: bool = False
    sample_dialog: List[str] = field(
        default_factory=lambda: [
            "2025-02-01 10:00 Alice scheduled the system upgrade.",
            "2025-02-01 14:00 Bob confirmed the patch deployment.",
            "2025-02-02 09:15 Alice reported latency improvements.",
        ]
    )

    def prepare(self) -> None:
        self.prepared = True

    def run(self, pipeline) -> BenchmarkResult:
        if not self.prepared:
            self.prepare()

        joined = "\n".join(self.sample_dialog)
        pipeline.ingest(joined, topic="operations")

        query = QueryContext(
            text="When did Alice report latency improvements?",
            topic="operations",
        )
        results = pipeline.retrieve(query)
        metrics = {
            "temporal_accuracy": float(bool(results)),
            "token_cost": float(len(results)),
        }
        return BenchmarkResult(name=self.name, metrics=metrics, notes="LoCoMo prototype run")

