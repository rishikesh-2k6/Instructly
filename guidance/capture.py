"""Window screenshot capture.

Windows-only. Uses PIL.ImageGrab over mss: ImageGrab.grab() returns a
PIL.Image directly, matching this module's required return type with no
conversion step, whereas mss would add a dependency purely to duplicate
what ImageGrab already does for a single, one-off capture (mss earns its
keep for high-frequency/live capture, which isn't what this session needs).

Both are screen-region grabs, not true off-screen window captures: the
target window must actually be visible on screen (not minimized, and not
fully occluded) for the capture to show real content — GDI's BitBlt, which
ImageGrab uses under the hood, reads whatever is currently composited on
screen at that region, not the window's private off-screen bitmap. In
practice this is fine for OpenShot's normal Qt UI; a window doing
exclusive-fullscreen DirectX rendering underneath other windows is the one
case where this approach could grab black/stale pixels instead.
"""

from __future__ import annotations

import ctypes
import warnings
from ctypes import wintypes
from typing import Optional

import win32gui
from PIL import Image, ImageGrab

# DWMWA_EXTENDED_FRAME_BOUNDS gives the window's true visible bounding
# rectangle. Plain GetWindowRect includes several pixels of invisible
# resize-border padding that Windows 10/11 adds around top-level windows,
# which would otherwise offset every captured screenshot (and, later, every
# annotation drawn on it) from what perception's UI Automation bounding
# boxes report — UI Automation bounding rectangles reflect the visible
# frame, not the padded one.
_DWMWA_EXTENDED_FRAME_BOUNDS = 9


def _find_window(window_title_substring: str) -> Optional[int]:
    """Return the hwnd of the first visible top-level window whose title
    contains `window_title_substring` (case-insensitive), or None."""
    target = window_title_substring.lower()
    matches: list[int] = []

    def _enum_handler(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and target in title.lower():
            matches.append(hwnd)

    win32gui.EnumWindows(_enum_handler, None)
    return matches[0] if matches else None


def _frame_bounds(hwnd: int) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) screen-absolute bounds for `hwnd`,
    preferring the DWM extended frame (excludes invisible resize padding),
    falling back to GetWindowRect if DWM isn't available."""
    rect = wintypes.RECT()
    try:
        result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            wintypes.DWORD(_DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if result == 0:  # S_OK
            return rect.left, rect.top, rect.right, rect.bottom
    except OSError:
        pass
    return win32gui.GetWindowRect(hwnd)


def get_window_bounds(window_title_substring: str) -> Optional[tuple[int, int, int, int]]:
    """Return (left, top, right, bottom) screen-absolute bounds of the first
    visible window matching `window_title_substring`, or None if no window
    matches.

    Exposed separately from capture_window_screenshot so callers (see
    guidance/render.py) can convert screen-absolute coordinates — such as
    perception's UI Automation bounding boxes — into coordinates relative
    to the captured image, using the same origin the capture itself used.
    """
    hwnd = _find_window(window_title_substring)
    if hwnd is None:
        return None
    return _frame_bounds(hwnd)


def capture_window_screenshot(window_title_substring: str) -> Image.Image:
    """Capture a screenshot of the window whose title contains
    `window_title_substring`.

    Captures just that window's on-screen bounds when a matching visible
    window is found. Falls back to a full-screen capture (emitting a
    warning) if no window matches, since a guidance screenshot showing the
    wrong window is worse than one that's simply larger than necessary.
    """
    bounds = get_window_bounds(window_title_substring)
    if bounds is None:
        warnings.warn(
            f"No visible window found matching {window_title_substring!r}; "
            "falling back to a full-screen capture.",
            stacklevel=2,
        )
        return ImageGrab.grab()

    return ImageGrab.grab(bbox=bounds)
