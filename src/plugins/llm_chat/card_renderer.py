"""Shared Pillow card layout. Pure rendering: no LLM calls, network or delivery."""

from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime
from io import BytesIO
from pathlib import Path
import re
import uuid
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
from .paths import data_path


@dataclass
class CardBlock:
    heading: str
    entries: list[str] = field(default_factory=list)
    metadata: list[str] = field(default_factory=list)
    accent: tuple = (47, 139, 123)
    body_color: tuple = (54, 76, 71)
    image: Image.Image | None = None
    body_size: int = 26
    entry_lines: int = 16


@dataclass
class CardDocument:
    title: str
    label: str
    blocks: list[CardBlock]
    subtitle: str = ""
    activity: str = ""
    footer: str = ""


def render_document(document: CardDocument) -> Image.Image:
    """Measure and draw once, using the same wrapping and spacing for every template."""
    if len(document.blocks) > 30:
        raise ValueError("too_many_card_blocks")
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    content_width = SVG_WIDTH - 2 * SVG_HORIZONTAL_PADDING
    left, right = SVG_HORIZONTAL_PADDING - 8, SVG_WIDTH - SVG_HORIZONTAL_PADDING + 8
    text_left, text_width = left + 28, right - left - 56

    def lines(text, size, width, maximum, bold=False):
        return _clamp_wrapped_lines(wrap_card_text(str(text)[:6000], _get_font(size, bold), width, measure), maximum)

    labels = lines(document.label, 20, content_width, 2)
    titles = lines(document.title, 48, content_width, 3, True)
    subtitles = lines(document.subtitle, 26, content_width, 3)
    activity = lines(document.activity, 20, content_width, 2)
    footer = lines(document.footer, 18, content_width, 2)
    header_height = len(labels) * 30 + 20 + len(titles) * 60 + 12 + len(subtitles) * 36 + 22 + len(activity) * 30 + 24
    prepared = []
    for block in document.blocks:
        headings = lines(block.heading, 30, text_width - 76, 3, True)
        body = [
            line for entry in block.entries for line in lines(entry, block.body_size, text_width, block.entry_lines)
        ]
        meta = [line for entry in block.metadata for line in lines(entry, 19, text_width, 3)]
        thumbnail = (
            ImageOps.contain(block.image, (text_width, 600), Image.Resampling.LANCZOS).convert("RGBA")
            if block.image
            else None
        )
        heading_height = max(38, len(headings) * 40)
        height = 22 + heading_height + 12 + len(body) * (block.body_size + 16) + 26
        if thumbnail is not None:
            height += thumbnail.height + 16
        if meta:
            height += 14 + len(meta) * 29
        prepared.append((block, headings, body, meta, thumbnail, heading_height, height))
    height = (
        SVG_TOP_PADDING
        + header_height
        + sum(p[-1] for p in prepared)
        + max(0, len(prepared) - 1) * SECTION_GAP
        + SVG_BOTTOM_PADDING
        + len(footer) * 26
    )
    if height > 16000:
        raise ValueError("card_too_tall")
    image = _paint_editorial_background(height)
    draw = ImageDraw.Draw(image)
    y = SVG_TOP_PADDING
    for text in labels:
        draw.text((SVG_HORIZONTAL_PADDING, y), text, font=_get_font(20), fill=TEXT_MUTED)
        y += 30
    y += 20
    for text in titles:
        draw.text((SVG_HORIZONTAL_PADDING, y), text, font=_get_font(48, True), fill=TEXT_PRIMARY)
        y += 60
    y += 12
    for text in subtitles:
        draw.text((SVG_HORIZONTAL_PADDING, y), text, font=_get_font(26), fill=TEXT_SECONDARY)
        y += 36
    _draw_accent_rule(draw, y + 6)
    y += 22
    for text in activity:
        draw.text((SVG_HORIZONTAL_PADDING, y), text, font=_get_font(20), fill=TEXT_ACCENT)
        y += 30
    y += 24
    for index, (block, headings, body, meta, thumbnail, heading_height, height) in enumerate(prepared, 1):
        _draw_translucent_panel(image, (left, y, right, y + height), accent=block.accent)
        draw = ImageDraw.Draw(image)
        _draw_pill(
            draw,
            (text_left, y + 23),
            f"{index:02d}",
            _get_font(17, True),
            outline=block.accent,
            text_fill=block.accent,
            padding_x=12,
            padding_y=5,
        )
        for i, text in enumerate(headings):
            draw.text((text_left + 76, y + 20 + i * 40), text, font=_get_font(30, True), fill=TEXT_PRIMARY)
        text_y = y + 22 + heading_height + 12
        if thumbnail is not None:
            image.alpha_composite(thumbnail, (text_left + (text_width - thumbnail.width) // 2, text_y))
            text_y += thumbnail.height + 16
            draw = ImageDraw.Draw(image)
        for text in body:
            draw.text((text_left, text_y), text, font=_get_font(block.body_size), fill=block.body_color)
            text_y += block.body_size + 16
        if meta:
            text_y += 14
        for text in meta:
            draw.text((text_left, text_y), text, font=_get_font(19), fill=TEXT_MUTED)
            text_y += 29
        y += height + SECTION_GAP
    for text in footer:
        width = draw.textlength(text, font=_get_font(18))
        draw.text((SVG_WIDTH - SVG_HORIZONTAL_PADDING - width, y + 8), text,
                  font=_get_font(18), fill=TEXT_MUTED)
        y += 26
    return image.convert("RGB")


def card_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def save_card_image(image: Image.Image, directory: Path, prefix: str, group_id: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_{int(group_id)}_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}.png"
    path.write_bytes(card_png(image))
    return path


SVG_WIDTH = 1080
SVG_HORIZONTAL_PADDING = 72
SVG_TOP_PADDING = 72
SVG_BOTTOM_PADDING = 72
SECTION_GAP = 20
# Mineral-white paper, ink-green typography and restrained crystal accents.
ACCENT_CYAN = (47, 139, 123)
ACCENT_BLUE = (78, 119, 143)
ACCENT_VIOLET = (127, 118, 155)
ACCENT_CORAL = (153, 115, 123)
TEXT_PRIMARY = (28, 57, 53)
TEXT_SECONDARY = (54, 76, 71)
TEXT_MUTED = (96, 115, 109)
TEXT_ACCENT = ACCENT_CYAN
TEXT_TOPIC = TEXT_SECONDARY
TEXT_MEMBER = TEXT_SECONDARY
TEXT_SUMMARY = TEXT_PRIMARY
RUNTIME_FONT_DIR = data_path("fonts")
FONT_CANDIDATES = [
    str(RUNTIME_FONT_DIR / "msyh.ttc"),
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
FONT_BOLD_CANDIDATES = [
    str(RUNTIME_FONT_DIR / "msyhbd.ttc"),
    "C:/Windows/Fonts/msyhbd.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def wrap_text_pixel(text: str, font, max_pixel_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    """按真实字体像素宽度换行，避免和画布宽度脱节导致文字挤在一侧。"""
    if not text:
        return []

    def _measure(s: str) -> int:
        if not s:
            return 0
        bbox = draw.textbbox((0, 0), s, font=font)
        return bbox[2] - bbox[0]

    wrapped_lines: list[str] = []
    for raw_line in str(text).splitlines() or [str(text)]:
        line = raw_line.strip()
        if not line:
            wrapped_lines.append("")
            continue

        current = ""
        for char in line:
            candidate = current + char
            if current and _measure(candidate) > max_pixel_width:
                wrapped_lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            wrapped_lines.append(current)

    # 避免中文标点单独落在新行开头，卡片里会显得像排版断裂。
    no_line_start = set("，。！？；：、）》】」』…—,.!?;:")
    normalized: list[str] = []
    for line in wrapped_lines:
        if normalized and line and line[0] in no_line_start:
            normalized[-1] += line[0]
            line = line[1:]
        if line or not normalized:
            normalized.append(line)

    return normalized or [""]


@lru_cache(maxsize=32)
def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载适合中文显示的字体。"""
    preferred_fonts = FONT_CANDIDATES[:]
    if bold:
        preferred_fonts = [*FONT_BOLD_CANDIDATES, *preferred_fonts]

    for font_path in preferred_fonts:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _paint_editorial_background(total_height: int) -> Image.Image:
    """Low-resolution aurora light, enlarged once; deterministic and inexpensive."""
    width, height = SVG_WIDTH // 4, max(1, total_height // 4)
    small = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(small)
    for y in range(height):
        t = y / max(1, height - 1)
        color = tuple(round(a * (1 - t) + b * t) for a, b in zip((220, 244, 238), (229, 223, 245)))
        draw.line((0, y, width, y), fill=color)
    glow = Image.new("RGBA", small.size)
    gd = ImageDraw.Draw(glow)
    for index, y in enumerate(range(-50, height, 210)):
        x = -65 if index % 2 == 0 else width - 130
        color = [(87, 215, 185, 120), (175, 140, 238, 100), (255, 212, 150, 95)][index % 3]
        gd.ellipse((x, y, x + 230, y + 200), fill=color)
    small = Image.alpha_composite(small.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(32)))
    image = small.resize((SVG_WIDTH, total_height), Image.Resampling.BICUBIC)
    facets = Image.new("RGBA", image.size)
    fd = ImageDraw.Draw(facets)
    for index, y in enumerate(range(0, total_height, 720)):
        x = 790 if index % 2 == 0 else -80
        fd.polygon([(x, y), (x + 255, y + 65), (x + 170, y + 360)], fill=(255, 255, 255, 65), outline=(255, 255, 255, 150), width=2)
        fd.polygon([(x + 255, y + 65), (x + 300, y + 300), (x + 170, y + 360)], fill=(116, 191, 198, 35))
        for dx, dy in [(40, 95), (210, 390)]:
            cx, cy = x + dx, y + dy
            fd.line((cx - 7, cy, cx + 7, cy), fill=(255, 255, 255, 200), width=2)
            fd.line((cx, cy - 7, cx, cy + 7), fill=(255, 255, 255, 200), width=2)
    image.alpha_composite(facets)
    return image


def _draw_translucent_panel(image, box, *, accent, radius=24):
    """Frosted background, a soft local shadow, and a luminous glass rim."""
    left, top, right, bottom = map(int, box)
    width, height = right - left, bottom - top
    shadow = Image.new("RGBA", (width + 32, height + 32))
    ImageDraw.Draw(shadow).rounded_rectangle((16, 16, width + 16, height + 16), radius=radius, fill=(*accent, 38))
    image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(9)), (left - 16, top - 8))
    glass = image.crop((left, top, right, bottom)).filter(ImageFilter.GaussianBlur(14))
    glass.alpha_composite(Image.new("RGBA", glass.size, (255, 255, 255, 182)))
    mask = Image.new("L", glass.size)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
    image.paste(glass, (left, top), mask)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box, radius=radius, outline=(255, 255, 255, 235), width=2)
    draw.line((left + radius, top + 2, right - radius, top + 2), fill=(255, 255, 255), width=2)
    draw.rounded_rectangle((left + 28, top + 16, left + 72, top + 19), radius=2, fill=accent)


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font,
    *,
    fill: tuple[int, int, int] = (236, 245, 238),
    outline: tuple[int, int, int] = (172, 206, 191),
    text_fill: tuple[int, int, int] = TEXT_ACCENT,
    padding_x: int = 15,
    padding_y: int = 7,
) -> tuple[int, int, int, int]:
    """绘制用于日期、统计或编号的小胶囊。"""
    x, y = position
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + padding_x * 2
    height = bbox[3] - bbox[1] + padding_y * 2
    pill_box = (x, y, x + width, y + height)
    draw.rounded_rectangle(
        pill_box,
        radius=10,
        fill=fill,
        outline=outline,
        width=1,
    )
    draw.text((x + padding_x, y + padding_y - bbox[1]), text, font=font, fill=text_fill)
    return pill_box


def _draw_accent_rule(draw: ImageDraw.ImageDraw, y: int) -> None:
    """绘制青—蓝—紫的短色带与延伸细线。"""
    x = SVG_HORIZONTAL_PADDING
    draw.rounded_rectangle((x, y, x + 76, y + 4), radius=2, fill=ACCENT_CYAN)
    draw.rounded_rectangle((x + 70, y, x + 150, y + 4), radius=2, fill=ACCENT_BLUE)
    draw.rounded_rectangle((x + 144, y, x + 224, y + 4), radius=2, fill=ACCENT_VIOLET)
    draw.line((x + 240, y + 2, SVG_WIDTH - SVG_HORIZONTAL_PADDING, y + 2), fill=(211, 227, 218), width=1)


def _clamp_wrapped_lines(lines: list[str], max_lines: int) -> list[str]:
    """限制卡片中的文字行数，并在截断时添加省略号。"""
    if len(lines) <= max_lines:
        return lines
    clamped = lines[:max_lines]
    last = clamped[-1].rstrip()
    clamped[-1] = (last[:-1] if len(last) > 1 else last) + "…"
    return clamped


def wrap_card_text(text: str, font, width: int, draw) -> list[str]:
    """Keep ordinary Latin words intact while wrapping mixed Chinese text."""
    lines, current = [], ""
    for token in re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]", text):
        if token == "\n":
            lines.append(current.rstrip())
            current = ""
            continue
        candidate = current + token
        box = draw.textbbox((0, 0), candidate, font=font)
        if current and box[2] - box[0] > width:
            lines.append(current.rstrip())
            current = token.lstrip()
        else:
            current = candidate
        # Very long identifiers still have to fit within the card.
        box = draw.textbbox((0, 0), current, font=font)
        if box[2] - box[0] > width:
            pieces = wrap_text_pixel(current, font, width, draw)
            lines.extend(pieces[:-1])
            current = pieces[-1]
    if current:
        lines.append(current.rstrip())
    return lines
