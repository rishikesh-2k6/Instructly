"""Draws a highlight annotation around a UI element on a screenshot.

Coordinates: `bounding_box` is expected relative to `image`'s own pixel
grid — (0, 0) is the image's top-left corner — NOT screen-absolute.
annotate_target only ever sees pixels within the image it's handed, with
no way to know where that image was captured from on screen, so it can't
do a screen-absolute conversion itself. Screen-absolute boxes (e.g. from
perception/matcher.py's UI Automation output) must be converted to
image-relative coordinates by the caller first; guidance/render.py does
this conversion using the window origin from guidance/capture.py.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

Box = tuple[int, int, int, int]

ACCENT_COLOR = (0, 209, 255, 255)  # bright cyan — reads against light and dark UIs
GLOW_COLOR = (0, 209, 255, 120)
SHADOW_COLOR = (0, 0, 0, 140)
CALLOUT_BG = (24, 24, 28, 235)
CALLOUT_TEXT_COLOR = (255, 255, 255, 255)

CORNER_RADIUS = 10
BORDER_WIDTH = 4
GLOW_BLUR_RADIUS = 8
GLOW_PADDING = 6
SHADOW_OFFSET = (2, 3)
SHADOW_BLUR_RADIUS = 3
CALLOUT_PADDING = (10, 6)
CALLOUT_CORNER_RADIUS = 6
CALLOUT_GAP = 10
EDGE_MARGIN = 8


def _load_font(size: int = 16) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("segoeui.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _expand_box(box: Box, amount: int) -> Box:
    left, top, right, bottom = box
    return left - amount, top - amount, right + amount, bottom + amount


def _draw_glow(overlay: Image.Image, box: Box) -> None:
    glow_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_box = _expand_box(box, GLOW_PADDING)
    glow_draw.rounded_rectangle(
        glow_box, radius=CORNER_RADIUS + GLOW_PADDING, outline=GLOW_COLOR, width=BORDER_WIDTH + 4
    )
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(GLOW_BLUR_RADIUS))
    overlay.alpha_composite(glow_layer)


def _draw_shadow(overlay: Image.Image, box: Box) -> None:
    shadow_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    dx, dy = SHADOW_OFFSET
    shadow_box = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
    shadow_draw.rounded_rectangle(shadow_box, radius=CORNER_RADIUS, outline=SHADOW_COLOR, width=BORDER_WIDTH)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))
    overlay.alpha_composite(shadow_layer)


def _draw_highlight_box(overlay: Image.Image, box: Box) -> None:
    ImageDraw.Draw(overlay).rounded_rectangle(box, radius=CORNER_RADIUS, outline=ACCENT_COLOR, width=BORDER_WIDTH)


def _callout_position(box: Box, callout_w: int, callout_h: int, image_size: tuple[int, int]) -> tuple[int, int]:
    image_width, image_height = image_size
    left, top, right, bottom = box

    # Prefer placing the callout above the box; fall back to below if that
    # would clip off the top edge.
    x = left
    y = top - callout_h - CALLOUT_GAP
    if y < EDGE_MARGIN:
        y = bottom + CALLOUT_GAP

    # Clamp on both axes so the callout is always fully within the image,
    # regardless of how close the target box is to any edge.
    x = max(EDGE_MARGIN, min(x, image_width - callout_w - EDGE_MARGIN))
    y = max(EDGE_MARGIN, min(y, image_height - callout_h - EDGE_MARGIN))
    return x, y


def _draw_callout(overlay: Image.Image, box: Box, label: str) -> None:
    draw = ImageDraw.Draw(overlay)
    font = _load_font()

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad_x, pad_y = CALLOUT_PADDING
    callout_w = text_w + pad_x * 2
    callout_h = text_h + pad_y * 2

    x, y = _callout_position(box, callout_w, callout_h, overlay.size)

    callout_box = (x, y, x + callout_w, y + callout_h)
    draw.rounded_rectangle(callout_box, radius=CALLOUT_CORNER_RADIUS, fill=CALLOUT_BG)
    draw.text((x + pad_x - text_bbox[0], y + pad_y - text_bbox[1]), label, font=font, fill=CALLOUT_TEXT_COLOR)


def annotate_target(image: Image.Image, bounding_box: Box, label: Optional[str] = None) -> Image.Image:
    """Return a copy of `image` with a highlight drawn around `bounding_box`.

    `bounding_box` is (left, top, right, bottom), relative to `image`'s own
    pixel grid. The highlight is a rounded rectangle in a bright accent
    color with a soft glow and drop shadow so it stays legible against any
    background. If `label` is given, a tooltip-style callout is drawn near
    the box, clamped to always stay fully within the image.
    """
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    _draw_glow(overlay, bounding_box)
    _draw_shadow(overlay, bounding_box)
    _draw_highlight_box(overlay, bounding_box)
    if label:
        _draw_callout(overlay, bounding_box, label)

    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")
