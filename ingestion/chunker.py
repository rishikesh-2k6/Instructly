"""Extracts text from a PDF and splits it into sections at numbered headings.

The heading pattern is configurable per-document: it was validated against
OpenShot's official documentation (116 sections detected with
DEFAULT_HEADING_PATTERN), but other apps' docs may use a different
numbering convention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# Matches numbered headings like "1.6.7 Properties" on their own line:
# digits, then one or more ".digit" groups (optionally trailing dot), then
# a title starting with a capital letter. Both `number` and `title` named
# groups are required by split_into_chunks.
#
# The minimum of two numeric levels (X.Y) is deliberate: a single-level
# minimum ((?:\d+\.)+\d*) also matches numbered UI-glossary list items like
# "1. Track Head" inside a section body, misdetecting them as new section
# headings. Real numbered headings in OpenShot's docs are always X.Y or
# deeper, so requiring >=2 levels avoids that collision.
DEFAULT_HEADING_PATTERN = r"^(?P<number>\d+(?:\.\d+)+)\.?\s+(?P<title>[A-Z][^\n]{0,120})$"


@dataclass
class Chunk:
    section_number: str
    title: str
    body_text: str


def extract_text(pdf_path: str | Path) -> str:
    """Extract raw text from a PDF, one page's text per line join."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def split_into_chunks(text: str, heading_pattern: str = DEFAULT_HEADING_PATTERN) -> list[Chunk]:
    """Split `text` into chunks at heading matches.

    `heading_pattern` must define `number` and `title` named groups, each
    matched against a single line (used with re.MULTILINE).
    """
    pattern = re.compile(heading_pattern, re.MULTILINE)
    matches = list(pattern.finditer(text))

    chunks: list[Chunk] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chunks.append(
            Chunk(
                section_number=match.group("number").rstrip("."),
                title=match.group("title").strip(),
                body_text=body,
            )
        )
    return chunks


def chunk_pdf(pdf_path: str | Path, heading_pattern: str = DEFAULT_HEADING_PATTERN) -> list[Chunk]:
    """Extract and chunk a PDF in one step."""
    text = extract_text(pdf_path)
    return split_into_chunks(text, heading_pattern)
