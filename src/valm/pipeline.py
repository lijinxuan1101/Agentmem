from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .anchor import AnchorGenerator
from .config import (
    EntropyConfig,
    HyperTokenizerConfig,
    OCRConfig,
    RetrievalConfig,
    VisualizerConfig,
)
from .entropy import SemanticEntropyEstimator
from .hyper_token import HyperTokenizer
from .injection import HyperTokenInjector
from .retrieval import VALMRetrievalEngine
from .state.controller import ConstantStateController
from .store import VALMStore
from .types import HyperToken, QueryContext, VisualToken
from .visualizer import VisualSketchBuffer


@dataclass
class VALMPipeline:
    visualizer: VisualSketchBuffer
    anchor_generator: AnchorGenerator
    entropy_estimator: SemanticEntropyEstimator
    hyper_tokenizer: HyperTokenizer
    store: VALMStore
    retriever: VALMRetrievalEngine
    injector: HyperTokenInjector
    state_controller: ConstantStateController

    @classmethod
    def build(
        cls,
        *,
        visualizer_config: VisualizerConfig | None = None,
        ocr_config: OCRConfig | None = None,
        entropy_config: EntropyConfig | None = None,
        hyper_config: HyperTokenizerConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> "VALMPipeline":
        visualizer_config = visualizer_config or VisualizerConfig()
        ocr_config = ocr_config or OCRConfig()
        entropy_config = entropy_config or EntropyConfig()
        hyper_config = hyper_config or HyperTokenizerConfig()
        retrieval_config = retrieval_config or RetrievalConfig()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if visualizer_config.vist_device == "auto":
            visualizer_config.vist_device = device
        if ocr_config.device == "auto":
            ocr_config.device = device

        visualizer = VisualSketchBuffer(visualizer_config)
        anchor_generator = AnchorGenerator(ocr_config=ocr_config)
        entropy_estimator = SemanticEntropyEstimator(entropy_config)
        hyper_tokenizer = HyperTokenizer(entropy_estimator, hyper_config)
        store = VALMStore()
        retriever = VALMRetrievalEngine(store, retrieval_config)
        injector = HyperTokenInjector()
        state_controller = ConstantStateController()
        return cls(
            visualizer=visualizer,
            anchor_generator=anchor_generator,
            entropy_estimator=entropy_estimator,
            hyper_tokenizer=hyper_tokenizer,
            store=store,
            retriever=retriever,
            injector=injector,
            state_controller=state_controller,
        )

    @classmethod
    def build_default(cls) -> "VALMPipeline":
        return cls.build()

    def ingest(self, text: str, *, topic: str | None = None) -> List[HyperToken]:
        tokens = self.visualizer.render(text)
        self.store.add_tokens(tokens, topic=topic)
        self.store.index_raw_text(text, topic)
        anchors = self.anchor_generator.generate(tokens)
        self.store.add_anchors(anchors)

        hyper_tokens: List[HyperToken] = []
        anchor_map = {anchor.token_id: anchor for anchor in anchors}
        for token in tokens:
            anchor = anchor_map.get(token.token_id)
            encoded = self.hyper_tokenizer.encode(token, anchor)
            if encoded is None:
                continue
            hyper_tokens.append(encoded)

        self.store.add_hyper_tokens(hyper_tokens)
        return hyper_tokens

    def retrieve(self, query: QueryContext) -> List[HyperToken]:
        ranked = self.retriever.retrieve(query)
        return [hyper for hyper, _ in ranked]

    def prepare_kv(self, hyper_tokens: List[HyperToken]):
        new_kv = self.injector.prepare_kv(hyper_tokens)
        self.state_controller.update(new_kv)
        return new_kv
