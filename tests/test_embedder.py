from unittest.mock import MagicMock

import pytest

from ingestion import embedder


def test_embed_text_returns_vector_matching_dimension(monkeypatch):
    monkeypatch.setattr(embedder, "_embed", MagicMock(return_value=[0.1] * embedder.EMBEDDING_DIM))

    vector = embedder.embed_text("some text")

    assert len(vector) == embedder.EMBEDDING_DIM


def test_embed_text_raises_loudly_on_dimension_mismatch(monkeypatch):
    monkeypatch.setattr(embedder, "_embed", MagicMock(return_value=[0.1] * 123))

    with pytest.raises(embedder.EmbeddingDimensionError):
        embedder.embed_text("some text")
