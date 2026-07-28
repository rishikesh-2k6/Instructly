"""Detects the currently focused window and returns its UI tree, so callers
don't strictly need to know the app name in advance.
"""

from __future__ import annotations

from typing import Optional

import uiautomation as auto

from perception.uia_tree import _walk


def get_active_window_ui_tree(max_depth: int = 6) -> Optional[tuple[str, list[dict]]]:
    """Return (window_title, ui_tree) for whichever top-level window
    currently has keyboard focus, or None if that can't be determined.

    Walks the tree directly from the resolved top-level control rather
    than re-searching by title through uia_tree.get_ui_tree — titles
    aren't guaranteed unique (e.g. two File Explorer windows can share a
    title), so re-searching could silently walk the wrong window.
    """
    try:
        focused = auto.GetFocusedControl()
        if focused is None:
            return None

        top_level = focused.GetTopLevelControl()
        if top_level is None:
            return None

        title = top_level.Name or ""
        if not title:
            return None

        return title, _walk(top_level, max_depth)
    except Exception:
        return None
