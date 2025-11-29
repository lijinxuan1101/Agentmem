from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ConstantStateController:
    """
    Manages a fixed-size latent state vector for MEM1-style processing.
    
    This controller maintains:
    1. A 'goal state' vector that persists across long horizons.
    2. A 'working memory' buffer that is updated via gated injection.
    3. Mechanisms to evict least-relevant information based on entropy/saliency.
    """
    
    state_dim: int = 768
    max_slots: int = 32
    decay_rate: float = 0.01
    
    _goal_state: List[float] = field(default_factory=list)
    _working_memory: List[Dict[str, float]] = field(default_factory=list)  # List of slots with values and metadata
    
    def initialize(self, goal_embedding: List[float]) -> None:
        """Sets the persistent goal state."""
        if len(goal_embedding) != self.state_dim:
            # In a real implementation, we might project or pad
            pass
        self._goal_state = goal_embedding
        self._working_memory = []

    def update(self, new_kvs: Dict[str, List[float]]) -> None:
        """
        Updates working memory with new key-value pairs from HyperTokens.
        
        Args:
            new_kvs: Output from HyperTokenInjector.prepare_kv()
        """
        # Parse gates and values
        gates = new_kvs.get("gates", [])
        # values = new_kvs.get("v", []) # Simplified: just tracking gates/slots for logic demo
        token_ids = new_kvs.get("token_ids", [])
        
        for i, token_id in enumerate(token_ids):
            gate = gates[i] if i < len(gates) else 0.5
            
            # Eviction policy: if full, remove lowest score slot
            if len(self._working_memory) >= self.max_slots:
                self._evict()
                
            self._working_memory.append({
                "id": token_id,
                "score": gate,
                "age": 0.0
            })
            
        # Apply time decay to existing slots
        for slot in self._working_memory:
            slot["score"] *= (1.0 - self.decay_rate)
            slot["age"] += 1.0

    def _evict(self) -> None:
        """Removes the memory slot with the lowest utility score."""
        if not self._working_memory:
            return
        # Find min score
        min_idx = 0
        min_score = self._working_memory[0]["score"]
        for i, slot in enumerate(self._working_memory):
            if slot["score"] < min_score:
                min_score = slot["score"]
                min_idx = i
        self._working_memory.pop(min_idx)

    def get_state_representation(self) -> Dict[str, Any]:
        """Returns the current compressed state for visualization or downstream usage."""
        return {
            "goal_initialized": bool(self._goal_state),
            "memory_usage": len(self._working_memory) / self.max_slots,
            "top_slots": sorted(self._working_memory, key=lambda x: x["score"], reverse=True)[:5]
        }


