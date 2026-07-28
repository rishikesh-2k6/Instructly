"""Type-specific structured extraction, matching the four payload shapes
documented in sql/001_init.sql. Each content_type gets its own explicit
prompt template rather than one generic prompt — local models follow
short, concrete instructions more reliably than long, nuanced ones.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ingestion.chunker import Chunk
from ingestion.ollama_client import generate_json

MAX_BODY_CHARS = 4000


# --- payload schemas, matching sql/001_init.sql exactly -------------------


class ProceduralStep(BaseModel):
    instruction: str
    ui_hint: dict = Field(default_factory=dict)


class ProceduralPayload(BaseModel):
    goal: str
    steps: list[ProceduralStep]
    prerequisites: list[str] = Field(default_factory=list)


class UIGlossaryPayload(BaseModel):
    element_name: str
    context: str
    description: str


class ReferenceProperty(BaseModel):
    name: str
    description: str
    range: Optional[str] = None


class ReferencePayload(BaseModel):
    entity_name: str
    properties: list[ReferenceProperty]


class ConceptualPayload(BaseModel):
    summary: str


class EmbeddingTitle(BaseModel):
    goal_title: str


# --- prompts, one per content_type -----------------------------------------

PROCEDURAL_PROMPT = """Extract the procedure described in this documentation section into structured steps.

Section title: {title}

Section text:
\"\"\"
{body}
\"\"\"

Return ONLY JSON with this exact shape:
{{
  "goal": "<short phrase describing what the user accomplishes>",
  "steps": [
    {{"instruction": "<one user action, e.g. 'Click File > Export Project'>", "ui_hint": {{}}}}
  ],
  "prerequisites": ["<anything the user must have or do first>"]
}}
Use an empty object {{}} for ui_hint on every step. List steps in the order they must be performed.
If there are no prerequisites, use an empty list []."""

UI_GLOSSARY_PROMPT = """This documentation section names and describes a single UI element or panel.

Section title: {title}

Section text:
\"\"\"
{body}
\"\"\"

Return ONLY JSON with this exact shape:
{{
  "element_name": "<name of the UI element/panel>",
  "context": "<where this element appears, e.g. 'Main toolbar' or 'Properties panel'>",
  "description": "<what the element does>"
}}"""

REFERENCE_PROMPT = """This documentation section is a reference table of properties, parameters, or settings.

Section title: {title}

Section text:
\"\"\"
{body}
\"\"\"

Return ONLY JSON with this exact shape:
{{
  "entity_name": "<name of the thing these properties belong to>",
  "properties": [
    {{"name": "<property name>", "description": "<what it does>", "range": "<valid range/values, or null if not applicable>"}}
  ]
}}"""

CONCEPTUAL_PROMPT = """Summarize this documentation section in 1-2 sentences.

Section title: {title}

Section text:
\"\"\"
{body}
\"\"\"

Return ONLY JSON with this exact shape:
{{"summary": "<1-2 sentence summary>"}}"""

GOAL_TITLE_PROMPT = """Write a short phrase (5-10 words) describing what this documentation section is about, suitable for a search index.

Section title: {title}

Section text:
\"\"\"
{body}
\"\"\"

Return ONLY JSON: {{"goal_title": "<short phrase>"}}"""


_EXTRACTORS: dict[str, tuple[str, type[BaseModel]]] = {
    "procedural": (PROCEDURAL_PROMPT, ProceduralPayload),
    "ui_glossary": (UI_GLOSSARY_PROMPT, UIGlossaryPayload),
    "reference": (REFERENCE_PROMPT, ReferencePayload),
    "conceptual": (CONCEPTUAL_PROMPT, ConceptualPayload),
}


def extract_payload(chunk: Chunk, content_type: str) -> dict:
    """Extract the type-specific structured payload for `chunk`."""
    if content_type not in _EXTRACTORS:
        raise ValueError(f"Unknown content_type: {content_type!r}")
    prompt_template, schema = _EXTRACTORS[content_type]
    prompt = prompt_template.format(title=chunk.title, body=chunk.body_text[:MAX_BODY_CHARS])
    result = generate_json(prompt, schema)
    return result.model_dump()


def extract_goal_title(chunk: Chunk) -> str:
    """Generate the short goal/title string that gets embedded (not the raw payload)."""
    prompt = GOAL_TITLE_PROMPT.format(title=chunk.title, body=chunk.body_text[:MAX_BODY_CHARS])
    result = generate_json(prompt, EmbeddingTitle)
    return result.goal_title
