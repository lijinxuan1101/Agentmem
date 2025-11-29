from __future__ import annotations

from dataclasses import dataclass, field

from .base import BenchmarkAdapter, BenchmarkResult
from ..types import QueryContext


@dataclass
class AgentBenchAdapter(BenchmarkAdapter):
    """Prototype adapter for tool-use / planning heavy tasks."""

    name: str = "agentbench"
    prepared: bool = False
    workflow_log: str = field(
        default_factory=lambda: "\n".join(
            [
                "Step 1: Open browser and navigate to invoices dashboard.",
                "Step 2: Download invoice 8821.",
                "Step 3: Verify payment status with API token AB12.",
                "Step 4: Update CRM with confirmation.",
            ]
        )
    )

    def prepare(self) -> None:
        self.prepared = True

    def run(self, pipeline) -> BenchmarkResult:
        if not self.prepared:
            self.prepare()

        pipeline.ingest(self.workflow_log, topic="ops")
        query = QueryContext(
            text="Which step requires API token usage?",
            topic="ops",
        )
        results = pipeline.retrieve(query)
        kv = pipeline.prepare_kv(results[:1]) if results else {"gates": []}
        metrics = {
            "tool_success": float(bool(results)),
            "kv_gate_strength": sum(abs(g) for g in kv.get("gates", [])),
        }
        return BenchmarkResult(name=self.name, metrics=metrics, notes="AgentBench prototype run")

