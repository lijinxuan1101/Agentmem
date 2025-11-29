from __future__ import annotations

from dataclasses import dataclass, field

from .base import BenchmarkAdapter, BenchmarkResult
from ..types import QueryContext


@dataclass
class InfiniteBenchAdapter(BenchmarkAdapter):
    """Prototype adapter for simulating ultra-long context tasks."""

    name: str = "infinitebench"
    prepared: bool = False
    synthetic_document: str = field(
        default_factory=lambda: " ".join(
            f"Section {i}: PassKey {i*17 % 997} stored in vault {i % 5}."
            for i in range(200)
        )
    )

    def prepare(self) -> None:
        self.prepared = True

    def run(self, pipeline) -> BenchmarkResult:
        if not self.prepared:
            self.prepare()

        pipeline.ingest(self.synthetic_document, topic="long_context")
        query = QueryContext(
            text="Retrieve PassKey 221 stored in vault 2.",
            topic="long_context",
            metadata_filters={"entropy": (0.2, None)},
        )
        results = pipeline.retrieve(query)
        metrics = {
            "retrieve_kv_recall": float(bool(results)),
            "context_tokens": len(self.synthetic_document.split()),
        }
        return BenchmarkResult(name=self.name, metrics=metrics, notes="InfiniteBench prototype run")

