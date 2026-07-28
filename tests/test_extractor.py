from unittest.mock import MagicMock

import pytest

from ingestion import extractor
from ingestion.chunker import Chunk


def make_chunk() -> Chunk:
    return Chunk(
        section_number="1.6.7",
        title="Export Settings",
        body_text="Resolution: the output video resolution. Range: 480p-4K.",
    )


def test_extract_payload_procedural(monkeypatch):
    payload = extractor.ProceduralPayload(
        goal="Export a video",
        steps=[extractor.ProceduralStep(instruction="Click File > Export Project", ui_hint={})],
        prerequisites=[],
    )
    monkeypatch.setattr(extractor, "generate_json", MagicMock(return_value=payload))

    result = extractor.extract_payload(make_chunk(), "procedural")

    assert result["goal"] == "Export a video"
    assert result["steps"][0]["instruction"] == "Click File > Export Project"
    assert result["prerequisites"] == []


def test_extract_payload_ui_glossary(monkeypatch):
    payload = extractor.UIGlossaryPayload(
        element_name="Timeline", context="Main window", description="Where clips are arranged."
    )
    monkeypatch.setattr(extractor, "generate_json", MagicMock(return_value=payload))

    result = extractor.extract_payload(make_chunk(), "ui_glossary")

    assert result["element_name"] == "Timeline"


def test_extract_payload_reference(monkeypatch):
    payload = extractor.ReferencePayload(
        entity_name="Export Settings",
        properties=[extractor.ReferenceProperty(name="Resolution", description="Output resolution", range="480p-4K")],
    )
    monkeypatch.setattr(extractor, "generate_json", MagicMock(return_value=payload))

    result = extractor.extract_payload(make_chunk(), "reference")

    assert result["entity_name"] == "Export Settings"
    assert result["properties"][0]["range"] == "480p-4K"


def test_extract_payload_reference_range_defaults_to_none():
    payload = extractor.ReferencePayload(
        entity_name="Export Settings",
        properties=[extractor.ReferenceProperty(name="Codec", description="Video codec")],
    )
    assert payload.properties[0].range is None


def test_extract_payload_conceptual(monkeypatch):
    payload = extractor.ConceptualPayload(summary="OpenShot is a free, open-source video editor.")
    monkeypatch.setattr(extractor, "generate_json", MagicMock(return_value=payload))

    result = extractor.extract_payload(make_chunk(), "conceptual")

    assert result == {"summary": "OpenShot is a free, open-source video editor."}


def test_extract_payload_unknown_content_type_raises():
    with pytest.raises(ValueError):
        extractor.extract_payload(make_chunk(), "not_a_real_type")


def test_extract_goal_title(monkeypatch):
    monkeypatch.setattr(
        extractor, "generate_json", MagicMock(return_value=extractor.EmbeddingTitle(goal_title="Export video settings"))
    )

    result = extractor.extract_goal_title(make_chunk())

    assert result == "Export video settings"
