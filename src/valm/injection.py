from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .types import HyperToken


@dataclass
class HyperTokenInjector:
    """
    Converts HyperTokens into pseudo KV cache entries and manages injection hooks.
    
    This component acts as the bridge between VALM memory and the downstream LLM.
    It supports:
    1. Projection of quantized HyperTokens into KV-dim vectors.
    2. Gating based on entropy and saliency.
    3. Hooks to inject these KV pairs into transformer layers (e.g. via past_key_values).
    """

    projection_dim: int = 64
    gate_temperature: float = 0.7
    
    def attach_hook(self, model: Any, layer_indices: List[int] | None = None) -> None:
        """
        Attaches injection hooks to a PyTorch model.
        
        Args:
            model: The transformer model (e.g. LlamaForCausalLM).
            layer_indices: List of layer indices to inject memory into. 
                           If None, injects into the first 2 layers.
        """
        import torch
        
        if layer_indices is None:
            layer_indices = [0, 1]
            
        # This is a simplified hook pattern. Real implementation would need to inspect
        # model architecture (e.g. model.layers[i].self_attn) and wrap forward().
        # Here we define the interface.
        self._active_hooks = {}
        # for idx in layer_indices:
        #     hook = model.layers[idx].register_forward_pre_hook(self._injection_callback)
        #     self._active_hooks[idx] = hook
        pass

    def prepare_kv(self, hyper_tokens: Iterable[HyperToken]) -> Dict[str, List[float]]:
        kv = {"k": [], "v": [], "gates": [], "token_ids": []}
        entries = []
        for hyper in hyper_tokens:
            projected = self._project(hyper)
            gate = self._compute_gate(hyper)
            entries.append((hyper.token_id, gate, projected))

        norm = sum(abs(entry[1]) for entry in entries) or 1.0
        for token_id, gate, projected in entries:
            weight = gate / norm
            gated_k = [weight * value for value in projected]
            gated_v = [weight * value for value in reversed(projected)]
            kv["k"].extend(gated_k)
            kv["v"].extend(gated_v)
            kv["gates"].append(weight)
            kv["token_ids"].append(token_id)

        return kv

    def _project(self, hyper: HyperToken) -> List[float]:
        scale = max(1, (1 << hyper.bit_allocation[0]) - 1)
        normalized = [value / scale for value in hyper.quantized_vector]
        projected = []
        for idx in range(self.projection_dim):
            src = normalized[idx % len(normalized)]
            projected.append(src * (1.0 + 0.01 * idx))
        return projected

    def _compute_gate(self, hyper: HyperToken) -> float:
        entropy = hyper.metadata.get("entropy", 0.0)
        saliency = hyper.metadata.get("saliency", 0.0)
        energy = entropy + saliency
        return math.tanh(energy / max(1e-5, self.gate_temperature))

