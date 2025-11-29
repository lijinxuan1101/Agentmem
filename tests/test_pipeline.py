from __future__ import annotations

import unittest

from valm.pipeline import VALMPipeline
from valm.types import QueryContext


SAMPLE_TEXT = """
Q1 2025 revenue for VisionEdge reached $123M, a 45% YoY growth.
The board meeting on Feb 2nd discussed merging the DeepSeek OCR stack
with the in-house VIST renderer. CTO Alice Zhang noted latency dropped
to 120ms after deploying the adaptive RVQ codec across all agents.
"""


class TestVALMPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = VALMPipeline.build_default()

    def test_ingest_and_retrieve(self) -> None:
        hyper_tokens = self.pipeline.ingest(SAMPLE_TEXT, topic="finance")
        self.assertTrue(hyper_tokens)

        query = QueryContext(text="What did Alice Zhang report about latency?", topic="finance")
        retrieved = self.pipeline.retrieve(query)
        self.assertTrue(retrieved)

        kv = self.pipeline.prepare_kv(retrieved[:2])
        self.assertIn("k", kv)
        self.assertIn("v", kv)
        self.assertEqual(len(kv["k"]), len(kv["v"]))
        self.assertIn("gates", kv)
        self.assertEqual(len(kv["gates"]), min(2, len(retrieved[:2])))

    def test_retrieve_with_metadata_filters_and_time(self) -> None:
        self.pipeline.ingest(SAMPLE_TEXT, topic="finance")
        anchor = next(iter(self.pipeline.store.anchors.values()))

        query = QueryContext(
            text="Who mentioned the adaptive RVQ codec?",
            topic="finance",
            metadata_filters={"entropy": (0.05, None)},
            time_hint=anchor.timestamp,
            time_window=600.0,
        )
        retrieved = self.pipeline.retrieve(query)
        self.assertTrue(retrieved)
        kv = self.pipeline.prepare_kv(retrieved)
        self.assertTrue(all(abs(gate) <= 1 for gate in kv["gates"]))


if __name__ == "__main__":
    unittest.main()

