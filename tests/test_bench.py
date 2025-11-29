from __future__ import annotations

import unittest

from valm.bench import AgentBenchAdapter, InfiniteBenchAdapter, LoCoMoAdapter
from valm.pipeline import VALMPipeline


class TestBenchAdapters(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = VALMPipeline.build_default()

    def test_locomo_adapter(self) -> None:
        adapter = LoCoMoAdapter()
        result = adapter.run(self.pipeline)
        self.assertIn("temporal_accuracy", result.metrics)

    def test_infinitebench_adapter(self) -> None:
        adapter = InfiniteBenchAdapter()
        result = adapter.run(self.pipeline)
        self.assertIn("retrieve_kv_recall", result.metrics)

    def test_agentbench_adapter(self) -> None:
        adapter = AgentBenchAdapter()
        result = adapter.run(self.pipeline)
        self.assertIn("kv_gate_strength", result.metrics)


if __name__ == "__main__":
    unittest.main()

