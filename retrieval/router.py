"""Organizes a single search() call's results by intent.

Single-shot: takes one query, returns one routed result. No "current step"
tracking or conversation memory — that belongs to chat orchestration,
built in a later session.
"""

from __future__ import annotations

from typing import Optional

from retrieval.search import search

# ui_glossary matches are always fetched alongside the primary result, for
# later use by the (not-yet-built) grounding module.
GLOSSARY_TOP_K = 3

# properties shown in a reference snippet, so it stays a short informational
# blurb rather than a full property dump.
MAX_SNIPPET_PROPERTIES = 5


def _format_reference_snippet(row: dict) -> str:
    payload = row["payload"]
    entity_name = payload.get("entity_name", row["title"])
    properties = payload.get("properties", [])[:MAX_SNIPPET_PROPERTIES]
    parts = [f"{prop['name']}: {prop['description']}" for prop in properties]
    return f"{entity_name} - " + "; ".join(parts) if parts else entity_name


def _format_primary(row: dict) -> dict:
    primary = {
        "content_type": row["content_type"],
        "title": row["title"],
        "source_section": row.get("source_section"),
        "grounding_confidence": row.get("grounding_confidence"),
        "similarity": row.get("similarity"),
    }

    if row["content_type"] == "procedural":
        payload = row["payload"]
        primary["goal"] = payload.get("goal")
        primary["steps"] = payload.get("steps", [])
        primary["prerequisites"] = payload.get("prerequisites", [])
    elif row["content_type"] == "reference":
        primary["snippet"] = _format_reference_snippet(row)
    else:
        primary["payload"] = row["payload"]

    return primary


def _format_glossary_entry(row: dict) -> dict:
    payload = row["payload"]
    return {
        "element_name": payload.get("element_name"),
        "context": payload.get("context"),
        "description": payload.get("description"),
        "title": row["title"],
        "source_section": row.get("source_section"),
        "similarity": row.get("similarity"),
    }


def route(query: str, app_name: str, top_k: int = 5) -> dict:
    """Search for `query` within `app_name` and organize results by intent.

    Returns:
        {"intent": "procedural" | "reference" | "unknown",
         "primary": dict | None,
         "glossary_context": list[dict]}
    """
    results = search(query, app_name, top_k=top_k)
    raw_glossary = search(query, app_name, top_k=GLOSSARY_TOP_K, content_types=["ui_glossary"])
    glossary_context = [_format_glossary_entry(row) for row in raw_glossary]

    if not results:
        return {"intent": "unknown", "primary": None, "glossary_context": glossary_context}

    top = results[0]
    intent = top["content_type"] if top["content_type"] in ("procedural", "reference") else "unknown"

    return {
        "intent": intent,
        "primary": _format_primary(top),
        "glossary_context": glossary_context,
    }
