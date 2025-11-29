from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VisualizerConfig:
    paragraph_tokens: int = 80
    max_canvas_chars: int = 4_000
    base_resolution: tuple[int, int] = (224, 224)
    render_mode: str = "vist"  # "vist" or "text"
    render_backend: str = "synthetic"  # "synthetic" or "hf_vit"
    wrap_width: int = 64
    patch_char_size: int = 16
    multiscale_levels: int = 2
    vist_model_name: str = "models/vit-base"
    vist_device: str = "auto"


@dataclass
class EntropyConfig:
    min_bit: int = 2
    max_bit: int = 12
    entropy_floor: float = 0.05
    entropy_ceiling: float = 1.5


@dataclass
class HyperTokenizerConfig:
    codebook_dim: int = 8
    stages: int = 4
    random_seed: int = 13
    drop_threshold: float = 0.15
    high_def_threshold: float = 0.6
    bit_boost: int = 2


@dataclass
class RetrievalConfig:
    coarse_top_k: int = 8
    residual_weight: float = 0.6
    metadata_weight: float = 0.4
    head_count: int = 3
    topic_boost: float = 0.15
    time_decay: float = 3600.0
    importance_threshold: float = 0.2


@dataclass
class OCRConfig:
    backend: str = "regex"  # "regex" or "deepseek"
    model_name: str = "models/deepseek-ocr"
    device: str = "auto"
    batch_size: int = 8
    max_new_tokens: int = 256
