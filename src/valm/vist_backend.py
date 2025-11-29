from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

from PIL import Image, ImageDraw, ImageFont

from .config import VisualizerConfig
from .vision import RenderedCanvas


class VISTPatch(Protocol):
    text: str
    embedding: List[float]
    saliency: float
    layout: tuple[int, int]
    resolution: tuple[int, int]


class VISTBackend(Protocol):
    def render(self, text: str, canvas_id: str | None = None) -> List[dict]:
        ...


def get_vist_backend(config: VisualizerConfig) -> Optional[VISTBackend]:
    if config.render_backend == "hf_vit":
        return HFViTBackend(config)
    if config.render_backend == "synthetic":
        return None
    raise ValueError(f"Unknown render_backend: {config.render_backend}")


@dataclass
class HFViTBackend:
    """Thin wrapper around a ViT encoder to mimic VIST features."""

    config: VisualizerConfig

    def __post_init__(self) -> None:
        try:
            import transformers
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for hf_vit render_backend. "
                "Install via `pip install transformers pillow`."
            ) from exc

        processor_cls = (
            getattr(transformers, "AutoImageProcessor", None)
            or getattr(transformers, "AutoFeatureExtractor", None)
            or getattr(transformers, "AutoProcessor", None)
        )

        self.processor = None
        self.processor_config = None
        if processor_cls is not None:
            try:
                self.processor = processor_cls.from_pretrained(self.config.vist_model_name, use_fast=True)
            except Exception:
                self.processor_config = self._load_preprocessor_config()
        else:
            self.processor_config = self._load_preprocessor_config()

        self.model = AutoModel.from_pretrained(self.config.vist_model_name)
        self.model.to(self.config.vist_device)
        self.model.eval()

    def render(self, text: str, canvas_id: str | None = None) -> List[dict]:
        canvas = self._render_canvas(text)
        image = self._canvas_to_image(canvas)

        import torch

        inputs = self._prepare_inputs(image)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Take CLS and patch embeddings as features
        patch_embeddings = outputs.last_hidden_state.squeeze(0).cpu().tolist()

        patches: List[dict] = []
        for idx, embedding in enumerate(patch_embeddings):
            if idx == 0:
                continue  # skip CLS
            patches.append(
                {
                    "text": text,
                    "embedding": embedding[:64],
                    "saliency": 0.5,
                    "layout": (idx % 16, idx // 16),
                    "resolution": self.config.base_resolution,
                }
            )

        return patches

    def _render_canvas(self, text: str) -> RenderedCanvas:
        from .vision import VISTLikeRenderer

        renderer = VISTLikeRenderer(self.config)
        return renderer.render_canvas(text)

    def _canvas_to_image(self, canvas: RenderedCanvas) -> Image.Image:
        # Try to load a font that supports bold/size if possible, else default
        # For saliency-aware rendering, we really need a truetype font to change size/weight
        # But default PIL font is bitmap and fixed size.
        try:
            # Try fetching a system font or a known path? 
            # In typical linux envs, DejaVuSans is common.
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            import os
            if os.path.exists(font_path):
                base_font = ImageFont.truetype(font_path, 15)
                bold_font = ImageFont.truetype(font_path, 18) # Bigger for saliency
            else:
                # Fallback to default if no font file found
                base_font = ImageFont.load_default()
                bold_font = base_font
        except Exception:
            base_font = ImageFont.load_default()
            bold_font = base_font

        try:
            line_height = getattr(base_font, "size", 15) + 4
        except AttributeError:
            line_height = 15
        
        width = self.config.wrap_width * 12 # Increased width multiplier for larger font safety
        height = max(1, len(canvas.lines)) * (line_height + 2)
        image = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(image)

        # Mock saliency for now since RenderedCanvas lines don't have per-line saliency yet.
        # In a real impl, `render_canvas` would return (text, saliency) tuples.
        # Here we just demo the capability:
        # If line starts with a number or capital, we treat it as "salient" (heuristic).
        
        y_cursor = 0
        for idx, line in enumerate(canvas.lines):
            is_salient = (len(line) > 0 and (line[0].isdigit() or line[0].isupper()))
            
            if is_salient and bold_font != base_font:
                font_to_use = bold_font
                fill_color = 0 # Black
            else:
                font_to_use = base_font
                fill_color = 100 if not is_salient else 0 # Gray if not salient
                
            draw.text((5, y_cursor), line, font=font_to_use, fill=fill_color)
            y_cursor += line_height

        return image.convert("RGB")

    def _load_preprocessor_config(self) -> dict | None:
        import json
        import os

        config_path = os.path.join(self.config.vist_model_name, "preprocessor_config.json")
        if not os.path.exists(config_path):
            return None
        with open(config_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def _prepare_inputs(self, image: Image.Image):
        import torch
        import numpy as np

        if self.processor is not None:
            return self.processor(images=image, return_tensors="pt").to(self.config.vist_device)

        size = self.processor_config.get("size", {}).get("shortest_edge", 224) if self.processor_config else 224
        image = image.resize((size, size))
        array = np.asarray(image).astype("float32") / 255.0
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        array = array[:, :, :3]
        mean = self.processor_config.get("image_mean", [0.5, 0.5, 0.5]) if self.processor_config else [0.5, 0.5, 0.5]
        std = self.processor_config.get("image_std", [0.5, 0.5, 0.5]) if self.processor_config else [0.5, 0.5, 0.5]
        array = (array - mean) / std
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.tensor(array).unsqueeze(0)
        return {"pixel_values": tensor.to(self.config.vist_device)}
