"""Render daily message counts and the most active member's QQ avatar."""
import asyncio
import math
from io import BytesIO

import httpx
from PIL import Image, ImageDraw, ImageOps

from .card_renderer import ACCENT_CYAN, ACCENT_VIOLET, TEXT_MUTED, TEXT_PRIMARY, _get_font

AVATAR_SIZE = 62
AVATAR_MAX_BYTES = 256 * 1024


async def fetch_leader_avatars(days, cache):
    """Fetch each distinct avatar once per briefing run."""
    leaders = {day["leader_id"] for day in days if day["leader_id"]}
    missing = leaders - cache.keys()
    if missing:
        async with httpx.AsyncClient(timeout=3) as client:
            async def fetch(user_id):
                avatar = None
                try:
                    async with asyncio.timeout(4), client.stream(
                        "GET", "https://q1.qlogo.cn/g", params={"b": "qq", "nk": user_id, "s": 100},
                    ) as response:
                        response.raise_for_status()
                        if response.headers.get("content-type", "").lower().startswith("image/"):
                            data = bytearray()
                            async for chunk in response.aiter_bytes():
                                data.extend(chunk)
                                if len(data) > AVATAR_MAX_BYTES:
                                    break
                            else:
                                avatar = bytes(data)
                except (TimeoutError, httpx.HTTPError):
                    pass
                cache[user_id] = avatar

            await asyncio.gather(*(fetch(user_id) for user_id in missing))
    return {user_id: cache[user_id] for user_id in leaders if cache.get(user_id)}


def _crown(draw, x, y):
    draw.polygon([(x-12, y+3), (x-6, y+7), (x, y), (x+6, y+7), (x+12, y+3),
                  (x+9, y+14), (x-9, y+14)], fill=(240, 199, 105), outline=(167, 114, 47))
    draw.line([(x-8, y+17), (x+8, y+17)], fill=(172, 120, 48), width=2)
    draw.ellipse((x-2, y+8, x+2, y+12), fill=(255, 241, 187))


def _avatar_thumbnail(raw):
    if not raw:
        return None
    try:
        with Image.open(BytesIO(raw)) as original:
            if max(original.size) > 1024:
                return None
            return ImageOps.fit(original.convert("RGB"), (AVATAR_SIZE, AVATAR_SIZE), Image.Resampling.LANCZOS)
    except (Image.DecompressionBombError, OSError, ValueError):
        return None


def render_activity_chart(days, avatars):
    chart = Image.new("RGBA", (896, 440))
    draw = ImageDraw.Draw(chart)

    def label(x, y, value, size, color=TEXT_PRIMARY, *, bold=False, anchor="lt"):
        draw.text((x, y), str(value), font=_get_font(size, bold), fill=color, anchor=anchor)

    label(4, 5, f"近 {len(days)} 日消息量", 25, bold=True)
    label(890, 8, f"累计 {sum(day['total'] for day in days):,} 条", 22, ACCENT_CYAN, anchor="rt")
    left, right, top, bottom = 85, 848, 132, 341
    peak = max(day["total"] for day in days)
    unit = 100 if peak >= 300 else 10 if peak >= 30 else 1
    step = max(1, math.ceil(peak / (3 * unit))) * unit
    ceiling = 3 * step

    def y_at(amount):
        return round(bottom - (bottom - top) * amount / ceiling)

    points = [(round(left + (right - left) * i / max(1, len(days) - 1)), y_at(day["total"]))
              for i, day in enumerate(days)]
    mask = Image.new("L", chart.size)
    ImageDraw.Draw(mask).polygon([*points, (right, bottom), (left, bottom)], fill=255)
    shade = Image.new("RGBA", chart.size)
    brush = ImageDraw.Draw(shade)
    for y in range(top, bottom + 1):
        opacity = round(66 * (bottom - y) / (bottom - top)) + 12
        brush.line([(left, y), (right, y)], fill=(84, 168, 146, opacity))
    chart.alpha_composite(Image.composite(shade, Image.new("RGBA", chart.size), mask))
    for tick in range(0, ceiling + 1, step):
        y = y_at(tick)
        draw.line([(left, y), (right, y)], fill=(128, 161, 153, 95), width=1)
        label(49, y - 8, f"{tick:,}", 16, TEXT_MUTED, anchor="rt")
    draw.line(points, fill=ACCENT_CYAN, width=4, joint="curve")

    clip = Image.new("L", (AVATAR_SIZE, AVATAR_SIZE))
    ImageDraw.Draw(clip).ellipse((0, 0, AVATAR_SIZE - 1, AVATAR_SIZE - 1), fill=255)
    for index, ((x, y), day) in enumerate(zip(points, days)):
        color = ACCENT_VIOLET if index == len(days) - 1 else ACCENT_CYAN
        leader = day.get("leader_id")
        label(x, y - (77 if leader else 24), f"{day['total']:,}", 21, color, bold=True, anchor="mt")
        if leader:
            _crown(draw, x, y - 53)
            draw.ellipse((x - 35, y - 35, x + 35, y + 35), fill=(251, 253, 250), outline=color, width=2)
            avatar = _avatar_thumbnail(avatars.get(leader))
            if avatar is not None:
                chart.paste(avatar, (x - AVATAR_SIZE // 2, y - AVATAR_SIZE // 2), clip)
                avatar.close()
            else:
                draw.ellipse((x - 25, y - 25, x + 25, y + 25), fill=(217, 232, 225))
            label(x, y + 43, f"{day['leader_count']:,}", 18, color, anchor="mt")
        else:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        label(x, 424, day["date"][5:].replace("-", "/"), 19, TEXT_MUTED, anchor="mb")
    return chart
