"""Embedding generation for knowledge_chunks rows."""

from __future__ import annotations

from ingestion.ollama_client import embed as _embed

# Must match sql/001_init.sql's `embedding vector(768)` column exactly.
# Confirmed empirically against the nomic-embed-text model — see
# sql/001_init.sql for details on what to do if OLLAMA_EMBED_MODEL changes.
EMBEDDING_DIM = 768


class EmbeddingDimensionError(RuntimeError):
    """Raised when the embedding model returns a vector of unexpected length."""


def embed_text(text: str) -> list[float]:
    """Generate an embedding for `text`, asserting it matches EMBEDDING_DIM.

    Fails loudly rather than letting a mismatched vector reach Supabase.
    """
    vector = _embed(text)
    if len(vector) != EMBEDDING_DIM:
        raise EmbeddingDimensionError(
            f"Expected embedding of dimension {EMBEDDING_DIM}, got {len(vector)}. "
            "OLLAMA_EMBED_MODEL likely doesn't match the model sql/001_init.sql's "
            "embedding column was sized for — update EMBEDDING_DIM and the schema "
            "together, and regenerate existing embeddings, if you're changing models."
        )
    return vector
