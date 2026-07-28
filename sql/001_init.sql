-- Instructly: initial schema
-- Run this in the Supabase SQL editor (or via `supabase db push` / psql)
-- against your project's Postgres database.

-- Enable pgvector for embedding similarity search.
create extension if not exists vector;

-- =============================================================================
-- Table: knowledge_chunks
--
-- One row per ingested documentation chunk for a target application (e.g.
-- "OpenShot"). `embedding` is computed from a short goal/title string
-- (via the local Ollama embedding model, OLLAMA_EMBED_MODEL), NOT from the
-- raw `payload` — payload holds the full structured content returned to the
-- user once a chunk is retrieved.
--
-- embedding dimension: 768
-- Confirmed empirically against the "nomic-embed-text" model installed via
-- Ollama on this machine (ollama pull nomic-embed-text; len(ollama.embed(...)
-- .embeddings[0]) == 768). If OLLAMA_EMBED_MODEL is ever changed to a model
-- with a different output dimension, this column's vector(768) size must be
-- updated to match EXACTLY, and knowledge_chunks must be recreated (existing
-- embeddings from the old model are not compatible with a new dimension or
-- a new embedding space, and must be regenerated too).
--
-- payload shapes (jsonb), by content_type:
--
--   content_type = 'procedural'
--     {
--       "goal": str,
--       "steps": [
--         { "instruction": str, "ui_hint": { ... } },
--         ...
--       ],
--       "prerequisites": [str, ...]
--     }
--
--   content_type = 'ui_glossary'
--     {
--       "element_name": str,
--       "context": str,
--       "description": str
--     }
--
--   content_type = 'reference'
--     {
--       "entity_name": str,
--       "properties": [
--         { "name": str, "description": str, "range": str | null },
--         ...
--       ]
--     }
--
--   content_type = 'conceptual'
--     {
--       "summary": str
--     }
-- =============================================================================
create table if not exists knowledge_chunks (
    id                   uuid primary key default gen_random_uuid(),
    app_name             text not null,
    content_type         text not null
                         check (content_type in ('procedural', 'ui_glossary', 'reference', 'conceptual')),
    grounding_confidence text
                         check (grounding_confidence in ('menu', 'canvas')),
    title                text not null,
    payload              jsonb not null,
    embedding            vector(768), -- nomic-embed-text via Ollama; embeds goal/title, not raw payload
    source_section       text,
    created_at           timestamptz not null default now()
);

create index if not exists knowledge_chunks_app_name_idx
    on knowledge_chunks (app_name);

create index if not exists knowledge_chunks_content_type_idx
    on knowledge_chunks (content_type);

-- Approximate nearest-neighbor index for cosine similarity search over
-- embedding. IVFFlat is fine to start with on a small/empty table; consider
-- switching to HNSW (pgvector >= 0.5, supported by Supabase) once chunk
-- volume is known.
create index if not exists knowledge_chunks_embedding_idx
    on knowledge_chunks
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- Helper RPC so a health check / test can confirm the pgvector extension is
-- enabled without raw SQL access (the PostgREST API used by supabase-py
-- only exposes tables/views/functions, not pg_catalog).
create or replace function pgvector_enabled()
returns boolean
language sql
stable
as $$
    select exists (select 1 from pg_extension where extname = 'vector');
$$;
