from unittest.mock import MagicMock

from retrieval import router


def make_procedural_row(similarity=0.9):
    return {
        "content_type": "procedural",
        "title": "Export a video",
        "source_section": "3.2",
        "grounding_confidence": "menu",
        "similarity": similarity,
        "payload": {
            "goal": "Export a video project to a file",
            "steps": [{"instruction": "Click File > Export Project", "ui_hint": {}}],
            "prerequisites": [],
        },
    }


def make_reference_row(similarity=0.85):
    return {
        "content_type": "reference",
        "title": "Export Settings",
        "source_section": "3.3",
        "grounding_confidence": None,
        "similarity": similarity,
        "payload": {
            "entity_name": "Export Settings",
            "properties": [
                {"name": "Resolution", "description": "Output resolution", "range": "480p-4K"},
                {"name": "Bitrate", "description": "Video bitrate", "range": "1-50 Mbps"},
            ],
        },
    }


def make_glossary_row(name="Timeline", similarity=0.7):
    return {
        "content_type": "ui_glossary",
        "title": name,
        "source_section": "1.3",
        "grounding_confidence": None,
        "similarity": similarity,
        "payload": {"element_name": name, "context": "Main window", "description": f"The {name} panel."},
    }


def make_conceptual_row(similarity=0.6):
    return {
        "content_type": "conceptual",
        "title": "About OpenShot",
        "source_section": "1.1",
        "grounding_confidence": None,
        "similarity": similarity,
        "payload": {"summary": "OpenShot is a free, open-source video editor."},
    }


def test_route_procedural_top_match_returns_full_ordered_steps(monkeypatch):
    def fake_search(query, app_name, top_k=5, content_types=None):
        if content_types == ["ui_glossary"]:
            return [make_glossary_row()]
        return [make_procedural_row(), make_reference_row()]

    monkeypatch.setattr(router, "search", fake_search)

    result = router.route("how do I export a video", "OpenShot")

    assert result["intent"] == "procedural"
    assert result["primary"]["content_type"] == "procedural"
    assert result["primary"]["goal"] == "Export a video project to a file"
    assert result["primary"]["steps"] == [{"instruction": "Click File > Export Project", "ui_hint": {}}]
    assert "snippet" not in result["primary"]


def test_route_reference_top_match_returns_snippet_not_steps(monkeypatch):
    def fake_search(query, app_name, top_k=5, content_types=None):
        if content_types == ["ui_glossary"]:
            return []
        return [make_reference_row()]

    monkeypatch.setattr(router, "search", fake_search)

    result = router.route("what does resolution mean", "OpenShot")

    assert result["intent"] == "reference"
    assert "steps" not in result["primary"]
    assert "goal" not in result["primary"]
    assert "Resolution: Output resolution" in result["primary"]["snippet"]


def test_route_glossary_always_fetched_regardless_of_primary_intent(monkeypatch):
    calls = []

    def fake_search(query, app_name, top_k=5, content_types=None):
        calls.append(content_types)
        if content_types == ["ui_glossary"]:
            return [make_glossary_row("Playhead")]
        return [make_procedural_row()]

    monkeypatch.setattr(router, "search", fake_search)

    result = router.route("how do I trim a clip", "OpenShot")

    assert ["ui_glossary"] in calls
    assert len(result["glossary_context"]) == 1
    assert result["glossary_context"][0]["element_name"] == "Playhead"


def test_route_glossary_entries_have_clean_shape(monkeypatch):
    def fake_search(query, app_name, top_k=5, content_types=None):
        if content_types == ["ui_glossary"]:
            return [make_glossary_row()]
        return [make_procedural_row()]

    monkeypatch.setattr(router, "search", fake_search)

    result = router.route("how do I trim a clip", "OpenShot")

    entry = result["glossary_context"][0]
    assert set(entry.keys()) == {"element_name", "context", "description", "title", "source_section", "similarity"}


def test_route_no_results_returns_unknown_with_glossary(monkeypatch):
    def fake_search(query, app_name, top_k=5, content_types=None):
        if content_types == ["ui_glossary"]:
            return [make_glossary_row()]
        return []

    monkeypatch.setattr(router, "search", fake_search)

    result = router.route("asdkjaslkdj nonsense query", "OpenShot")

    assert result["intent"] == "unknown"
    assert result["primary"] is None
    assert len(result["glossary_context"]) == 1


def test_route_conceptual_top_match_is_unknown_intent_but_still_surfaced(monkeypatch):
    def fake_search(query, app_name, top_k=5, content_types=None):
        if content_types == ["ui_glossary"]:
            return []
        return [make_conceptual_row()]

    monkeypatch.setattr(router, "search", fake_search)

    result = router.route("what is openshot", "OpenShot")

    assert result["intent"] == "unknown"
    assert result["primary"]["content_type"] == "conceptual"
    assert result["primary"]["payload"]["summary"] == "OpenShot is a free, open-source video editor."
