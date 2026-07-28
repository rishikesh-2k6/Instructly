"""Windows UI Automation tree extraction.

Uses the `uiautomation` package (a COM-based UI Automation wrapper) rather
than win32gui (which guidance/capture.py uses for simple window-rect
lookups) because it walks a live control tree with named properties
(Name, ControlTypeName, AutomationId, ClassName, BoundingRectangle,
IsEnabled) instead of raw window handles. It talks to whatever UI
Automation provider the target app implements (Win32, WPF, Qt, Electron,
...), so nothing here is specific to OpenShot.

NOTE: this project's perception session was originally meant to be built
from a real audit script + a live JSON run against OpenShot, to pin down
exact field shapes from observed data. That input was never provided, so
the field shapes below (control_type, name, automation_id, class_name,
bounding_box, is_enabled, depth) follow the names specified for this
module, using uiautomation's Control properties directly with no
reshaping. Treat as a reasonable default, not a verified-against-real-data
shape.
"""

from __future__ import annotations

from typing import Optional

import uiautomation as auto


def _find_top_level_window(window_title_substring: str) -> Optional[auto.Control]:
    """Return the first top-level window whose Name contains
    `window_title_substring` (case-insensitive), or None."""
    target = window_title_substring.lower()
    root = auto.GetRootControl()
    for child in root.GetChildren():
        name = child.Name or ""
        if target in name.lower():
            return child
    return None


def _control_to_dict(control: auto.Control, depth: int) -> dict:
    rect = control.BoundingRectangle
    return {
        "control_type": control.ControlTypeName,
        "name": control.Name,
        "automation_id": control.AutomationId,
        "class_name": control.ClassName,
        "bounding_box": (rect.left, rect.top, rect.right, rect.bottom),
        "is_enabled": control.IsEnabled,
        "depth": depth,
    }


def _walk(control: auto.Control, max_depth: int, depth: int = 0) -> list[dict]:
    """Flatten `control` and its descendants (up to `max_depth` levels
    below it) into a list of node dicts, depth-first.

    Individual controls that raise on property access or child
    enumeration (stale/disappearing COM elements — a real risk when
    walking a live, possibly-animating UI) are skipped rather than
    aborting the whole walk.
    """
    nodes: list[dict] = []
    try:
        nodes.append(_control_to_dict(control, depth))
    except Exception:
        return nodes

    if depth >= max_depth:
        return nodes

    try:
        children = control.GetChildren()
    except Exception:
        return nodes

    for child in children:
        nodes.extend(_walk(child, max_depth, depth + 1))

    return nodes


def get_ui_tree(window_title_substring: str, max_depth: int = 6) -> list[dict]:
    """Return a flattened UI Automation tree for the window whose title
    contains `window_title_substring`, up to `max_depth` levels below the
    window itself (which is depth 0).

    Each node: {control_type, name, automation_id, class_name,
    bounding_box, is_enabled, depth}. bounding_box is screen-absolute
    (left, top, right, bottom), matching what guidance/render.py expects.
    Returns an empty list if no matching window is found.
    """
    window = _find_top_level_window(window_title_substring)
    if window is None:
        return []
    return _walk(window, max_depth)
