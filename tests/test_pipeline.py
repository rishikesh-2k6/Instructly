import json
from unittest.mock import MagicMock

from ingestion import pipeline
from ingestion.chunker import Chunk


def make_chunks() -> list[Chunk]:
    return [
        Chunk(section_number="1.1", title="Intro", body_text="OpenShot is a free video editor."),
        Chunk(section_number="1.2", title="Trim a Clip", body_text="Drag the edge of a clip on the timeline."),
    ]


def test_process_chunk_wires_stages_together(monkeypatch):
    monkeypatch.setattr(pipeline, "classify_content_type", MagicMock(return_value="procedural"))
    monkeypatch.setattr(pipeline, "classify_grounding", MagicMock(return_value="canvas"))
    monkeypatch.setattr(
        pipeline, "extract_payload", MagicMock(return_value={"goal": "g", "steps": [], "prerequisites": []})
    )
    monkeypatch.setattr(pipeline, "extract_goal_title", MagicMock(return_value="Trim a clip"))
    monkeypatch.setattr(pipeline, "embed_text", MagicMock(return_value=[0.0] * 768))

    row = pipeline.process_chunk(make_chunks()[1], "OpenShot")

    assert row["app_name"] == "OpenShot"
    assert row["content_type"] == "procedural"
    assert row["grounding_confidence"] == "canvas"
    assert row["title"] == "Trim a Clip"
    assert row["source_section"] == "1.2"
    assert len(row["embedding"]) == 768


def test_process_chunk_skips_grounding_for_non_procedural(monkeypatch):
    monkeypatch.setattr(pipeline, "classify_content_type", MagicMock(return_value="conceptual"))
    monkeypatch.setattr(pipeline, "classify_grounding", MagicMock(side_effect=AssertionError("should not be called")))
    monkeypatch.setattr(pipeline, "extract_payload", MagicMock(return_value={"summary": "s"}))
    monkeypatch.setattr(pipeline, "extract_goal_title", MagicMock(return_value="Intro"))
    monkeypatch.setattr(pipeline, "embed_text", MagicMock(return_value=[0.0] * 768))

    row = pipeline.process_chunk(make_chunks()[0], "OpenShot")

    assert row["grounding_confidence"] is None


def test_run_dry_run_never_touches_supabase(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "chunk_pdf", MagicMock(return_value=make_chunks()))
    monkeypatch.setattr(
        pipeline,
        "process_chunk",
        MagicMock(
            return_value={
                "app_name": "OpenShot",
                "content_type": "conceptual",
                "grounding_confidence": None,
                "title": "Intro",
                "payload": {"summary": "s"},
                "embedding": [0.0] * 768,
                "source_section": "1.1",
            }
        ),
    )
    mock_create_client = MagicMock()
    monkeypatch.setattr(pipeline, "create_client", mock_create_client)
    monkeypatch.setattr(pipeline, "FAILED_CHUNKS_PATH", tmp_path / "failed_chunks.jsonl")

    pipeline.run(pdf_path="fake.pdf", app_name="OpenShot", dry_run=True)

    mock_create_client.assert_not_called()


def test_run_inserts_each_row_when_not_dry_run(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "chunk_pdf", MagicMock(return_value=make_chunks()))
    row = {
        "app_name": "OpenShot",
        "content_type": "conceptual",
        "grounding_confidence": None,
        "title": "Intro",
        "payload": {"summary": "s"},
        "embedding": [0.0] * 768,
        "source_section": "1.1",
    }
    monkeypatch.setattr(pipeline, "process_chunk", MagicMock(return_value=row))
    monkeypatch.setattr(pipeline.config, "require_env", MagicMock())
    mock_table = MagicMock()
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_table
    monkeypatch.setattr(pipeline, "create_client", MagicMock(return_value=mock_supabase))
    monkeypatch.setattr(pipeline, "FAILED_CHUNKS_PATH", tmp_path / "failed_chunks.jsonl")

    pipeline.run(pdf_path="fake.pdf", app_name="OpenShot", dry_run=False)

    assert mock_table.insert.call_count == len(make_chunks())


def test_run_logs_failed_chunks_and_continues(monkeypatch, tmp_path):
    chunks = make_chunks()
    monkeypatch.setattr(pipeline, "chunk_pdf", MagicMock(return_value=chunks))

    def fake_process_chunk(chunk, app_name):
        if chunk.section_number == "1.1":
            raise RuntimeError("model returned garbage after retries")
        return {
            "app_name": app_name,
            "content_type": "procedural",
            "grounding_confidence": "canvas",
            "title": chunk.title,
            "payload": {},
            "embedding": [0.0] * 768,
            "source_section": chunk.section_number,
        }

    monkeypatch.setattr(pipeline, "process_chunk", fake_process_chunk)
    failed_path = tmp_path / "failed_chunks.jsonl"
    monkeypatch.setattr(pipeline, "FAILED_CHUNKS_PATH", failed_path)

    pipeline.run(pdf_path="fake.pdf", app_name="OpenShot", dry_run=True)

    assert failed_path.exists()
    lines = failed_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["section_number"] == "1.1"
    assert "model returned garbage after retries" in record["reason"]


def test_run_respects_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "chunk_pdf", MagicMock(return_value=make_chunks()))
    mock_process_chunk = MagicMock(
        return_value={
            "app_name": "OpenShot",
            "content_type": "conceptual",
            "grounding_confidence": None,
            "title": "Intro",
            "payload": {"summary": "s"},
            "embedding": [0.0] * 768,
            "source_section": "1.1",
        }
    )
    monkeypatch.setattr(pipeline, "process_chunk", mock_process_chunk)
    monkeypatch.setattr(pipeline, "FAILED_CHUNKS_PATH", tmp_path / "failed_chunks.jsonl")

    pipeline.run(pdf_path="fake.pdf", app_name="OpenShot", limit=1, dry_run=True)

    assert mock_process_chunk.call_count == 1
