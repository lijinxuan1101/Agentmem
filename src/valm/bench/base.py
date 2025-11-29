from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline import VALMPipeline


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    metrics: Dict[str, float]
    notes: str | None = None


class BenchmarkAdapter(Protocol):
    name: str

    def prepare(self) -> None:
        ...

    def run(self, pipeline: "VALMPipeline") -> BenchmarkResult:
        ...

