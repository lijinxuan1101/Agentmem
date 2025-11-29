from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

from PIL import Image, ImageDraw, ImageFont

from .config import OCRConfig
from .types import Anchor, VisualToken


def _ensure_flash_attention_stub() -> None:
    """
    DeepSeek's transformers code expects `LlamaFlashAttention2` to exist.
    Older transformers releases do not ship it, so we provide a thin shim
    that simply falls back to the standard LlamaAttention implementation.
    """

    try:
        from transformers.models.llama import modeling_llama
    except Exception:
        return

    if hasattr(modeling_llama, "LlamaFlashAttention2"):
        return

    class LlamaFlashAttention2(modeling_llama.LlamaAttention):  # type: ignore
        def __init__(self, config, layer_idx: int):
            super().__init__(config=config, layer_idx=layer_idx)

        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)

    modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2  # type: ignore[attr-defined]


class OCRClient(Protocol):
    def extract(self, tokens: Iterable[VisualToken]) -> List[Anchor]:
        ...


def get_ocr_client(config: OCRConfig) -> OCRClient:
    if config.backend == "regex":
        return RegexOCRClient()
    if config.backend == "deepseek":
        return DeepSeekOCRClient(config)
    raise ValueError(f"Unknown OCR backend: {config.backend}")


@dataclass
class RegexOCRClient:
    """Fallback OCR implementation using heuristics."""

    def extract(self, tokens: Iterable[VisualToken]) -> List[Anchor]:
        from .anchor import _ENTITY_PATTERN
        import time
        import uuid

        base_time = time.time()
        anchors: List[Anchor] = []
        for idx, token in enumerate(tokens):
            matches = _ENTITY_PATTERN.findall(token.text_span)
            if not matches:
                continue
            anchors.append(
                Anchor(
                    anchor_id=f"an-{uuid.uuid4().hex[:8]}",
                    token_id=token.token_id,
                    text=" ".join(matches[:6]),
                    entities=matches[:10],
                    timestamp=base_time + idx * 60,
                    confidence=min(0.99, 0.5 + 0.5 * token.saliency),
                )
            )
        return anchors


@dataclass
class DeepSeekOCRClient:
    """Wraps DeepSeek-OCR style models via transformers or ONNX."""

    config: OCRConfig
    tokenizer: any = None
    processor: any = None

    def __post_init__(self) -> None:
        _ensure_flash_attention_stub()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for DeepSeek OCR backend. "
                "Install via `pip install transformers pillow`."
            ) from exc
        try:
            from transformers import AutoProcessor  # type: ignore
        except ImportError:
            AutoProcessor = None  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        try:
            if AutoProcessor is not None:
                self.processor = AutoProcessor.from_pretrained(self.config.model_name, trust_remote_code=True)
            else:
                self.processor = None
        except Exception:
            self.processor = None
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model.to(self.config.device)
        self.model.eval()

    def extract(self, tokens: Iterable[VisualToken]) -> List[Anchor]:
        import torch
        import uuid
        import time

        anchors: List[Anchor] = []
        token_list = list(tokens)
        total_tokens = len(token_list)
        if total_tokens == 0:
            return anchors

        font = self._load_font()
        prompt = "<image>\nConvert the document to markdown."
        batch_size = max(1, self.config.batch_size)
        start_time = time.time()

        for start in range(0, total_tokens, batch_size):
            end = min(total_tokens, start + batch_size)
            batch_tokens = token_list[start:end]
            batch_images = [self._render_token(token, font) for token in batch_tokens]
            try:
                batch_texts = self._run_batch(batch_images, prompt)
            except Exception as exc:  # pragma: no cover - defensive
                batch_texts = [f"OCR Error: {exc}"] * len(batch_tokens)

            for idx, (token, text) in enumerate(zip(batch_tokens, batch_texts)):
                anchors.append(
                    Anchor(
                        anchor_id=f"an-{uuid.uuid4().hex[:8]}",
                        token_id=token.token_id,
                        text=text,
                        entities=text.split()[:10],
                        timestamp=start_time + (start + idx) * 60.0,
                        confidence=0.9,
                    )
                )

            if total_tokens > 10:
                progress = min(total_tokens, end)
                print(f"  OCR processing: {progress}/{total_tokens} tokens...", end="\r", flush=True)

        if total_tokens > 10:
            print(f"  OCR processing: {total_tokens}/{total_tokens} tokens complete.{' ' * 10}")
        return anchors

    def _load_font(self) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            return ImageFont.load_default()

    def _render_token(self, token: VisualToken, font: ImageFont.ImageFont) -> Image.Image:
        width, height = token.resolution
        image = Image.new("L", (max(32, width), max(32, height)), color=255)
        draw = ImageDraw.Draw(image)
        draw.multiline_text((4, 4), token.text_span[:512], font=font, fill=0)
        return image.convert("RGB")

    def _run_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        import torch

        if hasattr(self.model, "generate") and self.processor is not None:
            texts = [prompt] * len(images)
            inputs = self.processor(images=images, text=texts, return_tensors="pt").to(self.config.device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
            tokenizer = getattr(self.processor, "tokenizer", None) or self.tokenizer
            return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        if hasattr(self.model, "infer"):
            return self._run_infer(images, prompt)

        return ["" for _ in images]

    def _run_infer(self, images: List[Image.Image], prompt: str) -> List[str]:
        import os
        import tempfile

        outputs: List[str] = []
        with tempfile.TemporaryDirectory() as tmp_out_dir:
            temp_paths: List[str] = []
            for idx, image in enumerate(images):
                tmp_path = os.path.join(tmp_out_dir, f"ocr_{idx}.jpg")
                image.save(tmp_path, format="JPEG")
                temp_paths.append(tmp_path)

            for path in temp_paths:
                try:
                    res = self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=path,
                        output_path=tmp_out_dir,
                        save_results=False,
                        eval_mode=True,
                    )
                    outputs.append(res if isinstance(res, str) else str(res))
                except Exception as err:
                    outputs.append(f"Inference Error: {err}")

        return outputs
