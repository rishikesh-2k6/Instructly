"""Smoke tests for the Supabase/pgvector schema and the local Ollama server.

These hit real services — they are not mocked. They require:
  - A .env with real SUPABASE_URL / SUPABASE_KEY values, with
    sql/001_init.sql already applied to that project.
  - A running local Ollama server (OLLAMA_HOST) with OLLAMA_MODEL and
    OLLAMA_EMBED_MODEL pulled.
"""

import ollama
import pytest
from supabase import Client, create_client

import config


@pytest.fixture(scope="module")
def supabase() -> Client:
    config.require_env()
    return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


@pytest.fixture(scope="module")
def ollama_client() -> ollama.Client:
    return ollama.Client(host=config.OLLAMA_HOST)


def test_knowledge_chunks_table_exists(supabase: Client) -> None:
    response = (
        supabase.table("knowledge_chunks")
        .select("id, app_name, content_type, grounding_confidence, title, payload, embedding, source_section")
        .limit(1)
        .execute()
    )
    assert response.data is not None


def test_pgvector_extension_enabled(supabase: Client) -> None:
    response = supabase.rpc("pgvector_enabled").execute()
    assert response.data is True


def test_ollama_server_reachable(ollama_client: ollama.Client) -> None:
    response = ollama_client.list()
    installed = {model.model for model in response.models}
    assert installed, "Ollama is reachable but has no models pulled."


def test_ollama_chat_model_responds(ollama_client: ollama.Client) -> None:
    response = ollama_client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": "Reply with the single word: OK"}],
    )
    assert response.message.content.strip() != ""


def test_ollama_embed_model_responds(ollama_client: ollama.Client) -> None:
    response = ollama_client.embed(model=config.OLLAMA_EMBED_MODEL, input="Trim a clip in the timeline")
    assert len(response.embeddings[0]) == 768
