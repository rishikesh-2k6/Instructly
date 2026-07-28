"""Orchestrates capture -> annotate -> save to a temp file."""

from __future__ import annotations

import os
import tempfile

from guidance.annotate import annotate_target
from guidance.capture import capture_window_screenshot, get_window_bounds

Box = tuple[int, int, int, int]


def render_guidance(window_title_substring: str, bounding_box: Box, label: str) -> str:
    """Capture `window_title_substring`, highlight `bounding_box`, save, and
    return the resulting PNG's file path.

    `bounding_box` is expected in SCREEN-ABSOLUTE coordinates — the same
    space perception's UI Automation bounding boxes use. This function
    converts it to coordinates relative to the captured image (using the
    window's on-screen origin from guidance/capture.py:get_window_bounds)
    before calling annotate_target, which only knows about pixels within
    the image it's given, not the window's position on screen. If the
    window can't be found, capture falls back to a full-screen grab, whose
    origin is (0, 0) — so no offset is applied in that case.
    """
    image = capture_window_screenshot(window_title_substring)
    bounds = get_window_bounds(window_title_substring)
    origin_x, origin_y = (bounds[0], bounds[1]) if bounds is not None else (0, 0)

    left, top, right, bottom = bounding_box
    relative_box = (left - origin_x, top - origin_y, right - origin_x, bottom - origin_y)

    annotated = annotate_target(image, relative_box, label)

    fd, path = tempfile.mkstemp(suffix=".png", prefix="instructly_guidance_")
    os.close(fd)
    annotated.save(path)
    return path
