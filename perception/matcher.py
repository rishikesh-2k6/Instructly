"""Matches a workflow step's ui_hint against a live UI Automation tree.

ui_hint shapes (as produced by a procedural workflow step's payload):
    {"type": "menu", "path": ["File", "Export Project"]}
    {"type": "dialog_field" | "button" | "unknown", "label": "Export Video"}

Matching priority:
    a. Exact name match (case-insensitive) against ui_tree entries.
    b. Fuzzy string match (rapidfuzz) above SIMILARITY_THRESHOLD.
    c. If glossary_context is given and (a) and (b) both fail: fuzzy-match
       the hint's target text against glossary entries first to resolve a
       canonical element_name, then retry (a)/(b) against the tree using
       that canonical name instead of the raw hint text.

For "menu" hints, only the last path segment is matched against the tree
(e.g. "Export Project" for ["File", "Export Project"]) — a static tree
snapshot generally won't show a submenu's items until the menu is opened,
so matching the full path would need live interaction (clicking through
each level), not just tree matching. That's out of scope here.
"""

from __future__ import annotations

from typing import Optional

from rapidfuzz import fuzz

SIMILARITY_THRESHOLD = 75.0  # rapidfuzz token_sort_ratio, 0-100
GLOSSARY_MATCH_CONFIDENCE_DISCOUNT = 0.9  # matched via glossary indirection, not the tree directly


def _hint_target_text(ui_hint: dict) -> Optional[str]:
    if ui_hint.get("type") == "menu":
        path = ui_hint.get("path") or []
        return path[-1] if path else None
    return ui_hint.get("label")


def _exact_match(target: str, ui_tree: list[dict]) -> Optional[dict]:
    target_lower = target.lower()
    for node in ui_tree:
        if (node.get("name") or "").lower() == target_lower:
            return node
    return None


def _best_fuzzy_match(target: str, ui_tree: list[dict]) -> tuple[Optional[dict], float]:
    best_node = None
    best_score = 0.0
    for node in ui_tree:
        name = node.get("name") or ""
        if not name:
            continue
        score = fuzz.token_sort_ratio(target, name)
        if score > best_score:
            best_score = score
            best_node = node
    return best_node, best_score


def _resolve_via_glossary(target: str, glossary_context: list[dict]) -> Optional[str]:
    """Fuzzy-match `target` against glossary entries' element_name, return
    the best-matching canonical element_name if it clears the threshold."""
    best_name = None
    best_score = 0.0
    for entry in glossary_context:
        element_name = entry.get("element_name") or ""
        if not element_name:
            continue
        score = fuzz.token_sort_ratio(target, element_name)
        if score > best_score:
            best_score = score
            best_name = element_name
    return best_name if best_score >= SIMILARITY_THRESHOLD else None


def _result(node: dict, confidence: float, method: str) -> dict:
    return {
        "bounding_box": node["bounding_box"],
        "name": node["name"],
        "confidence": confidence,
        "match_method": method,
    }


def find_element(
    ui_hint: dict,
    ui_tree: list[dict],
    glossary_context: Optional[list[dict]] = None,
) -> Optional[dict]:
    """Resolve `ui_hint` to a live control in `ui_tree`.

    Returns {"bounding_box", "name", "confidence", "match_method"} on
    success, or None if nothing clears the matching bar — including after
    a glossary-assisted retry, if `glossary_context` is given. Grounding
    failure is always signaled as None, never as a low-confidence guess;
    callers must handle None explicitly (see chat/session.py).
    """
    target = _hint_target_text(ui_hint)
    if not target:
        return None

    exact = _exact_match(target, ui_tree)
    if exact is not None:
        return _result(exact, 1.0, "exact")

    fuzzy_node, fuzzy_score = _best_fuzzy_match(target, ui_tree)
    if fuzzy_node is not None and fuzzy_score >= SIMILARITY_THRESHOLD:
        return _result(fuzzy_node, fuzzy_score / 100.0, "fuzzy")

    if glossary_context:
        canonical_name = _resolve_via_glossary(target, glossary_context)
        if canonical_name:
            exact2 = _exact_match(canonical_name, ui_tree)
            if exact2 is not None:
                return _result(exact2, GLOSSARY_MATCH_CONFIDENCE_DISCOUNT, "glossary_exact")

            fuzzy2_node, fuzzy2_score = _best_fuzzy_match(canonical_name, ui_tree)
            if fuzzy2_node is not None and fuzzy2_score >= SIMILARITY_THRESHOLD:
                return _result(
                    fuzzy2_node, (fuzzy2_score / 100.0) * GLOSSARY_MATCH_CONFIDENCE_DISCOUNT, "glossary_fuzzy"
                )

    return None
