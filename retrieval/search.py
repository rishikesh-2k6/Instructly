"""Vector similarity search over knowledge_chunks.

Usage:
    python -m retrieval.search "how do I export a video" --app-name OpenShot
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

from supabase import Client, create_client

import config
from ingestion.embedder import embed_text

_supabase: Optional[Client] = None


def _get_client() -> Client:
    global _supabase
    if _supabase is None:
        config.require_env()
        _supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    return _supabase


def search(
    query: str,
    app_name: str,
    top_k: int = 5,
    content_types: Optional[list[str]] = None,
) -> list[dict]:
    """Embed `query` and return the top_k most similar knowledge_chunks rows.

    Embedding uses ingestion/embedder.py (the same model + dimension check
    used at ingestion time), so query and stored vectors live in the same
    space. Each result dict has: id, app_name, content_type,
    grounding_confidence, title, payload, source_section, similarity
    (cosine similarity, higher is better).
    """
    embedding = embed_text(query)
    supabase = _get_client()
    response = supabase.rpc(
        "match_knowledge_chunks",
        {
            "query_embedding": embedding,
            "match_app_name": app_name,
            "match_content_types": content_types,
            "match_count": top_k,
        },
    ).execute()
    return response.data


def _print_routed_result(routed: dict) -> None:
    intent = routed["intent"]
    primary = routed["primary"]

    print(f"intent: {intent}\n")

    if primary is None:
        print("No matches found for this app_name.")
    elif primary["content_type"] == "procedural":
        print(f"[{primary['title']}]  similarity={primary['similarity']:.3f}  grounding={primary['grounding_confidence']}")
        print(f"goal: {primary['goal']}")
        if primary["prerequisites"]:
            print("prerequisites:")
            for prereq in primary["prerequisites"]:
                print(f"  - {prereq}")
        print("steps:")
        for i, step in enumerate(primary["steps"], start=1):
            print(f"  {i}. {step['instruction']}")
    elif primary["content_type"] == "reference":
        print(f"[{primary['title']}]  similarity={primary['similarity']:.3f}")
        print(primary["snippet"])
    else:
        print(f"[{primary['title']}] ({primary['content_type']})  similarity={primary['similarity']:.3f}")
        print(json.dumps(primary["payload"], indent=2))

    print(f"\nui_glossary context ({len(routed['glossary_context'])}):")
    if not routed["glossary_context"]:
        print("  (none)")
    for entry in routed["glossary_context"]:
        print(f"  - {entry['element_name']} ({entry['context']}): {entry['description']}  [similarity={entry['similarity']:.3f}]")


def main() -> None:
    # Imported here, not at module level, to avoid a circular import with
    # retrieval.router (which imports `search` from this module).
    from retrieval.router import route

    parser = argparse.ArgumentParser(description="Query knowledge_chunks and print the routed result.")
    parser.add_argument("query", help='Natural-language query, e.g. "how do I export a video"')
    parser.add_argument("--app-name", required=True, help='Target app name, e.g. "OpenShot".')
    parser.add_argument("--top-k", type=int, default=5, help="Number of primary search results to consider.")
    args = parser.parse_args()

    routed = route(args.query, args.app_name, top_k=args.top_k)
    _print_routed_result(routed)


if __name__ == "__main__":
    main()
