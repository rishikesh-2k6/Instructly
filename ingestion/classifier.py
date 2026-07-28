"""Classifies documentation chunks by content_type, and by
grounding_confidence for procedural chunks, via the local Ollama chat model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ingestion.chunker import Chunk
from ingestion.ollama_client import generate_json

MAX_BODY_CHARS = 4000


class ContentTypeResult(BaseModel):
    content_type: Literal["procedural", "ui_glossary", "reference", "conceptual"]


class GroundingResult(BaseModel):
    grounding_confidence: Literal["menu", "canvas"]


CONTENT_TYPE_PROMPT = """You are classifying a section of software documentation into exactly one category.

Categories:
- procedural: describes a sequence of user actions toward a goal (clicking menus, buttons, dialogs, in order).
- ui_glossary: a numbered list or table naming UI elements/panels, usually tied to a screenshot with numbered callouts.
- reference: a table of properties, parameters, or settings and what they mean or their valid range. Not a sequence of actions.
- conceptual: general descriptive or background text that is not a set of steps and not a list of UI elements.

Section title: {title}

Section text:
\"\"\"
{body}
\"\"\"

Return ONLY JSON: {{"content_type": "<one of procedural, ui_glossary, reference, conceptual>"}}"""

GROUNDING_PROMPT = """You are classifying how a procedural (step-by-step) documentation section is grounded.

- "menu": every action follows an explicit, named path through menus, dialogs, or buttons (e.g. "File > Export Project").
- "canvas": some actions are free-form manipulation on a canvas, timeline, or preview area (dragging, resizing, clicking arbitrary points) with no fixed named path.

Section title: {title}

Section text:
\"\"\"
{body}
\"\"\"

Return ONLY JSON: {{"grounding_confidence": "<one of menu, canvas>"}}"""


def classify_content_type(chunk: Chunk) -> str:
    prompt = CONTENT_TYPE_PROMPT.format(title=chunk.title, body=chunk.body_text[:MAX_BODY_CHARS])
    result = generate_json(prompt, ContentTypeResult)
    return result.content_type


def classify_grounding(chunk: Chunk) -> str:
    prompt = GROUNDING_PROMPT.format(title=chunk.title, body=chunk.body_text[:MAX_BODY_CHARS])
    result = generate_json(prompt, GroundingResult)
    return result.grounding_confidence
