-- Instructly: retrieval RPC
-- Run this in the Supabase SQL editor (or via `supabase db push` / psql),
-- after 001_init.sql.
--
-- PostgREST (what supabase-py's .table()/.select() calls go through) has no
-- way to express a `<=>` vector distance operator or `order by` on it, so
-- similarity search goes through this RPC function instead, called via
-- supabase.rpc("match_knowledge_chunks", {...}) from retrieval/search.py.
--
-- `embedding <=> query_embedding` is cosine *distance* here because the
-- knowledge_chunks_embedding_idx index (sql/001_init.sql) uses
-- vector_cosine_ops. similarity = 1 - distance, so 1.0 is identical and 0.0
-- is orthogonal.
create or replace function match_knowledge_chunks(
    query_embedding vector(768),
    match_app_name text,
    match_content_types text[] default null,
    match_count int default 5
)
returns table (
    id uuid,
    app_name text,
    content_type text,
    grounding_confidence text,
    title text,
    payload jsonb,
    source_section text,
    similarity float
)
language sql
stable
as $$
    select
        knowledge_chunks.id,
        knowledge_chunks.app_name,
        knowledge_chunks.content_type,
        knowledge_chunks.grounding_confidence,
        knowledge_chunks.title,
        knowledge_chunks.payload,
        knowledge_chunks.source_section,
        1 - (knowledge_chunks.embedding <=> query_embedding) as similarity
    from knowledge_chunks
    where knowledge_chunks.app_name = match_app_name
      and (match_content_types is null or knowledge_chunks.content_type = any(match_content_types))
    order by knowledge_chunks.embedding <=> query_embedding
    limit match_count;
$$;
