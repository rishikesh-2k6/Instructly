from unittest.mock import MagicMock

from ingestion import classifier
from ingestion.chunker import Chunk


def make_chunk() -> Chunk:
    return Chunk(
        section_number="1.6.7",
        title="Trimming a Clip",
        body_text="Click and drag the edge of a clip on the timeline to trim it.",
    )


def test_classify_content_type(monkeypatch):
    mock_generate = MagicMock(return_value=classifier.ContentTypeResult(content_type="procedural"))
    monkeypatch.setattr(classifier, "generate_json", mock_generate)

    result = classifier.classify_content_type(make_chunk())

    assert result == "procedural"
    prompt_arg, schema_arg = mock_generate.call_args[0]
    assert "Trimming a Clip" in prompt_arg
    assert schema_arg is classifier.ContentTypeResult


def test_classify_grounding(monkeypatch):
    mock_generate = MagicMock(return_value=classifier.GroundingResult(grounding_confidence="canvas"))
    monkeypatch.setattr(classifier, "generate_json", mock_generate)

    result = classifier.classify_grounding(make_chunk())

    assert result == "canvas"
