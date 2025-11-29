"""
VALM (Vision-Anchored Latent Memory) package bootstrap.

The package provides building blocks to:
1. Render long text into synthetic visual tokens (visualizer).
2. Anchor those tokens back to textual spans (anchor).
3. Estimate semantic entropy and allocate compression bits (entropy).
4. Produce latent HyperTokens through a lightweight ARVQ-inspired codec (hyper_token).
5. Persist and retrieve HyperTokens with hybrid semantic/metadata filters (store, retrieval).
6. Inject retrieved HyperTokens into downstream models (injection).
"""

from .types import VisualToken, Anchor, HyperToken, QueryContext
from .visualizer import VisualSketchBuffer
from .anchor import AnchorGenerator
from .entropy import SemanticEntropyEstimator
from .hyper_token import HyperTokenizer
from .store import VALMStore
from .retrieval import VALMRetrievalEngine
from .injection import HyperTokenInjector
from .pipeline import VALMPipeline

__all__ = [
    "VisualToken",
    "Anchor",
    "HyperToken",
    "QueryContext",
    "VisualSketchBuffer",
    "AnchorGenerator",
    "SemanticEntropyEstimator",
    "HyperTokenizer",
    "VALMStore",
    "VALMRetrievalEngine",
    "HyperTokenInjector",
    "VALMPipeline",
]

