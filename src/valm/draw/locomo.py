from __future__ import annotations

"""
LoCoMo 专用渲染器

当前版本负责：
- 加载字体
- 计算画布大小
- 按行把 RenderedCanvas 文本画到图像上
- **对行内时间戳 (timestamp) 片段做高亮**：
  - 支持 ISO 日期（2023-05-08）
  - 支持 HH:MM / HH:MM:SS 时间
  - 支持 10 位 Unix 时间戳
  - 支持自然语言时间 (1:56 pm on 8 May, 2023)
"""

import math
import os
import re
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

from ..config import VisualizerConfig
from ..vision import RenderedCanvas


def _shannon_entropy(text: str) -> float:
    """
    无模型版 Shannon 熵，按词粒度近似衡量**单个词的平均信息量**。

    实现方式：
    - 先对一段文本做分词并统计词频，得到整段的 Shannon 熵 H = -∑ p_i log p_i
    - 再除以 token 数，得到「每个 token 的平均 self-information」
    """
    s = text.strip()
    if not s:
        return 0.0

    tokens = s.split()
    if not tokens:
        return 0.0

    freqs = Counter(tokens)
    total = float(len(tokens))
    H = 0.0
    for c in freqs.values():
        p = c / total
        H -= p * math.log(p + 1e-12)
    # 返回的是「每个 token 的平均信息量」
    return H / total


def _collect_token_stats(lines: list[str], dialog_id_pattern: re.Pattern[str]) -> tuple[Counter, dict[str, float], float, float]:
    """
    收集整张画布上的 token 频率和逐 token 信息量，用于做灰度映射。
    - 统计范围：去掉 D1:1 之类对话编号前缀后的内容
    - 信息量定义：I(token) = -log p(token)
    """
    token_freqs: Counter = Counter()
    for raw_line in lines:
        line = raw_line.rstrip("\n").rstrip()
        if not line:
            continue
        m = dialog_id_pattern.match(line)
        content = m.group(3) if m else line
        tokens = content.split()
        token_freqs.update(tokens)

    total_tokens = float(sum(token_freqs.values()))
    if total_tokens == 0:
        return token_freqs, {}, 0.0, 0.0

    token_info: dict[str, float] = {}
    info_vals: list[float] = []
    for tok, c in token_freqs.items():
        p = c / total_tokens
        info = -math.log(p + 1e-12)
        token_info[tok] = info
        info_vals.append(info)

    info_min = min(info_vals) if info_vals else 0.0
    info_max = max(info_vals) if info_vals else 0.0
    return token_freqs, token_info, info_min, info_max


def _draw_segment_with_token_gray(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    token_info: dict[str, float],
    info_min: float,
    info_max: float,
) -> int:
    """
    按 token 粒度绘制一段普通文本：
    - 每个 token 的灰度根据 I(token) = -log p(token) 映射
    - 空格等分隔符按当前 x 原样前移
    返回绘制结束后的 x 坐标。
    """
    if not text:
        return x

    # 简单按空格拆分，保留分隔符
    parts = re.split(r"(\s+)", text)
    for part in parts:
        if part == "":
            continue
        if part.isspace():
            # 空白：按宽度前移，但不绘制
            if hasattr(draw, "textlength"):
                x += int(draw.textlength(part, font=font))
            else:
                x += draw.textsize(part, font=font)[0]
            continue

        info = token_info.get(part, info_min)
        if info_max > info_min:
            score = (info - info_min) / (info_max - info_min)
        else:
            score = 0.0
        # 信息量高 → 更黑；低 → 更浅
        gray = int(160 - score * 100)
        gray = max(40, min(200, gray))
        color = (gray, gray, gray)

        draw.text((x, y), part, font=font, fill=color)
        if hasattr(draw, "textlength"):
            x += int(draw.textlength(part, font=font))
        else:
            x += draw.textsize(part, font=font)[0]

    return x


def render_locomo_canvas_to_image(config: VisualizerConfig, canvas: RenderedCanvas) -> Image.Image:
    """
    将 LoCoMo 的 RenderedCanvas 渲染成基础图像，
    对行内时间戳片段做高亮，并根据无模型 Shannon 熵分布其他文字的灰度。
    """
    # 1. 字体加载：直接使用 TrueType（假定环境已提供）
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    base_font = ImageFont.truetype(font_path, 16)

    line_height = base_font.size + 4

    # 2. 画布大小：简单按字符宽度估算
    horizontal_padding = 8
    width_chars = max(config.wrap_width, canvas.width)
    width = horizontal_padding * 2 + width_chars * 10
    height = max(1, len(canvas.lines)) * (line_height + 2) + 8

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 3. 模式定义
    # 3.1 时间戳：ISO 日期 / HH:MM(:SS) / Unix / 自然语言时间
    timestamp_pattern = re.compile(
        r"(\b\d{4}-\d{1,2}-\d{1,2}\b"  # 2023-05-08
        r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"  # 13:56 或 13:56:30
        r"|\b\d{10}\b"  # 10 位 Unix 时间戳
        r"|\(\s*\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+\d{1,2}\s+[A-Za-z]+,\s*\d{4}\s*\)"  # (1:56 pm on 8 May, 2023)
        r")",
        re.IGNORECASE,
    )
    # 3.2 对话编号：例如 "D1:1 " / "D23:17\t"
    dialog_id_pattern = re.compile(r"^(D\d+:\d+)(\s*)(.*)")

    # 4. 收集 token 级别的信息量，用于按 token 控制灰度
    token_freqs, token_info, info_min, info_max = _collect_token_stats(list(canvas.lines), dialog_id_pattern)

    # 5. 行级渲染（普通文本按 token 灰度 + D1:1 高亮 + 时间戳高亮，行内可有多个时间戳）
    y_cursor = 4
    ts_bg_color = (220, 235, 255)
    ts_text_color = (0, 0, 160)
    dialog_color = (128, 0, 128)  # D1:1 等对话编号用紫色区分

    for idx, raw_line in enumerate(canvas.lines):
        line = raw_line.rstrip("\n").rstrip()
        if not line:
            y_cursor += line_height
            continue

        x = horizontal_padding

        # 4.1 先处理对话编号前缀（例如 "D1:1 Caroline: ..."）
        dialog_match = dialog_id_pattern.match(line)
        if dialog_match:
            dialog_id = dialog_match.group(1)
            spacing = dialog_match.group(2) or " "
            rest = dialog_match.group(3)

            # 绘制 D1:1 本身（仅颜色区分，不加背景）
            draw.text(
                (x, y_cursor),
                dialog_id,
                font=base_font,
                fill=dialog_color,
            )
            if hasattr(draw, "textlength"):
                x += int(draw.textlength(dialog_id + spacing, font=base_font))
            else:
                x += draw.textsize(dialog_id + spacing, font=base_font)[0]

            content = rest
        else:
            # 没有对话编号时，整个行内容都参与后续时间戳高亮
            content = line

        # 使用 finditer 在整行中查找所有时间戳片段
        last_end = 0
        for match in timestamp_pattern.finditer(content):
            start, end = match.start(), match.end()

            # 1) 画上一个时间戳之后到当前时间戳之前的普通文本（按 token 灰度）
            if start > last_end:
                plain_text = content[last_end:start]
                x = _draw_segment_with_token_gray(
                    draw,
                    plain_text,
                    x=x,
                    y=y_cursor,
                    font=base_font,
                    token_info=token_info,
                    info_min=info_min,
                    info_max=info_max,
                )

            # 2) 画当前时间戳背景块 + 文本
            ts_text = content[start:end]
            if ts_text:
                if hasattr(draw, "textlength"):
                    ts_width = int(draw.textlength(ts_text, font=base_font))
                else:
                    ts_width = draw.textsize(ts_text, font=base_font)[0]

                draw.rectangle(
                    [
                        (x - 1, y_cursor - 1),
                        (x + ts_width + 1, y_cursor + line_height - 3),
                    ],
                    fill=ts_bg_color,
                )
                draw.text(
                    (x, y_cursor),
                    ts_text,
                    font=base_font,
                    fill=ts_text_color,
                )
                x += ts_width

            last_end = end

        # 3) 画最后一个时间戳之后剩余的普通文本（按 token 灰度）
        if last_end < len(content):
            tail_text = content[last_end:]
            x = _draw_segment_with_token_gray(
                draw,
                tail_text,
                x=x,
                y=y_cursor,
                font=base_font,
                token_info=token_info,
                info_min=info_min,
                info_max=info_max,
            )

        # 如果整行没有任何时间戳匹配，且没有对话编号前缀，则整体按 token 灰度渲染
        if last_end == 0 and not dialog_match:
            _ = _draw_segment_with_token_gray(
                draw,
                line,
                x=horizontal_padding,
                y=y_cursor,
                font=base_font,
                token_info=token_info,
                info_min=info_min,
                info_max=info_max,
            )

        y_cursor += line_height

    return image


