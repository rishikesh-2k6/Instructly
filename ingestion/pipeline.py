"""Ingestion pipeline: PDF -> chunks -> classify -> extract -> embed -> Supabase.

Usage:
    python -m ingestion.pipeline --pdf path/to/doc.pdf --app-name "OpenShot"
    python -m ingestion.pipeline --pdf path/to/doc.pdf --app-name "OpenShot" --limit 10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from supabase import Client, create_client

import config
from ingestion.chunker import DEFAULT_HEADING_PATTERN, Chunk, chunk_pdf
from ingestion.classifier import classify_content_type, classify_grounding
from ingestion.embedder import embed_text
from ingestion.extractor import extract_goal_title, extract_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FAILED_CHUNKS_PATH = Path("failed_chunks.jsonl")


def process_chunk(chunk: Chunk, app_name: str) -> dict:
    """Run one chunk through classify -> extract -> embed, returning a row dict."""
    content_type = classify_content_type(chunk)

    grounding_confidence = None
    if content_type == "procedural":
        grounding_confidence = classify_grounding(chunk)

    payload = extract_payload(chunk, content_type)
    goal_title = extract_goal_title(chunk)
    embedding = embed_text(goal_title)

    return {
        "app_name": app_name,
        "content_type": content_type,
        "grounding_confidence": grounding_confidence,
        "title": chunk.title,
        "payload": payload,
        "embedding": embedding,
        "source_section": chunk.section_number,
    }


def log_failed_chunk(chunk: Chunk, reason: str, path: Path) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "section_number": chunk.section_number,
                    "title": chunk.title,
                    "reason": reason,
                }
            )
            + "\n"
        )


def run(
    pdf_path: str,
    app_name: str,
    heading_pattern: str = DEFAULT_HEADING_PATTERN,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    logger.info("Extracting and chunking %s", pdf_path)
    chunks = chunk_pdf(pdf_path, heading_pattern)
    logger.info("Found %d sections", len(chunks))

    if limit is not None:
        chunks = chunks[:limit]
        logger.info("Limiting to first %d chunks", limit)

    supabase: Optional[Client] = None
    if not dry_run:
        config.require_env()
        supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

    succeeded = 0
    failed = 0
    for i, chunk in enumerate(chunks, start=1):
        logger.info("[%d/%d] %s %s", i, len(chunks), chunk.section_number, chunk.title)
        try:
            row = process_chunk(chunk, app_name)
        except Exception as exc:  # one bad chunk must not kill the whole run
            logger.error("Failed on section %s (%s): %s", chunk.section_number, chunk.title, exc)
            log_failed_chunk(chunk, str(exc), FAILED_CHUNKS_PATH)
            failed += 1
            continue

        if dry_run:
            preview = {**row, "embedding": f"<{len(row['embedding'])}-dim vector>"}
            logger.info("DRY RUN - would insert:\n%s", json.dumps(preview, indent=2))
        else:
            supabase.table("knowledge_chunks").insert(row).execute()

        succeeded += 1

    logger.info("Done. %d succeeded, %d failed.", succeeded, failed)
    if failed:
        logger.info("Failed chunks logged to %s", FAILED_CHUNKS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF doc into knowledge_chunks.")
    parser.add_argument("--pdf", required=True, help="Path to the source PDF.")
    parser.add_argument("--app-name", required=True, help='Target app name, e.g. "OpenShot".')
    parser.add_argument(
        "--heading-pattern",
        default=DEFAULT_HEADING_PATTERN,
        help="Regex for section headings, with `number` and `title` named groups (see ingestion/chunker.py).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N chunks.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be inserted without writing to Supabase.",
    )
    args = parser.parse_args()

    run(
        pdf_path=args.pdf,
        app_name=args.app_name,
        heading_pattern=args.heading_pattern,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
