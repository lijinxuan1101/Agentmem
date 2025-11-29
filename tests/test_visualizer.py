from __future__ import annotations

import unittest

from valm.config import VisualizerConfig
from valm.visualizer import VisualSketchBuffer


SAMPLE_TEXT = "VIST compression trims 120K tokens into 3K visual anchors with DeepSeek OCR guidance."


class TestVisualSketchBuffer(unittest.TestCase):
    def test_vist_mode_produces_tokens(self) -> None:
        buffer = VisualSketchBuffer(VisualizerConfig(render_mode="vist", wrap_width=32, patch_char_size=8))
        tokens = buffer.render(SAMPLE_TEXT)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(token.metadata.get("char_density") is not None for token in tokens))

    def test_text_mode_fallback(self) -> None:
        buffer = VisualSketchBuffer(VisualizerConfig(render_mode="text", paragraph_tokens=10))
        tokens = buffer.render(SAMPLE_TEXT)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(token.text_span for token in tokens))


if __name__ == "__main__":
    unittest.main()

