from unittest.mock import MagicMock

import pytest

from chat import session as session_module
from chat.session import GuidanceSession, ReplyClassification, _rule_based_classify
from ingestion.ollama_client import GenerationError


def make_procedural_routed(num_steps=2, grounding="menu"):
    steps = [{"instruction": f"Step {i + 1} instruction", "ui_hint": {"type": "button", "label": f"Btn {i + 1}"}}
             for i in range(num_steps)]
    return {
        "intent": "procedural",
        "primary": {
            "content_type": "procedural",
            "title": "Export a video",
            "goal": "Export your project to a video file",
            "steps": steps,
            "prerequisites": [],
            "grounding_confidence": grounding,
        },
        "glossary_context": [{"element_name": "Timeline", "context": "Main window", "description": "..."}],
    }


def make_reference_routed():
    return {
        "intent": "reference",
        "primary": {"content_type": "reference", "title": "Export Settings", "snippet": "Resolution: output size"},
        "glossary_context": [],
    }


def make_unknown_routed():
    return {"intent": "unknown", "primary": None, "glossary_context": []}


def mock_grounding_success(monkeypatch, bounding_box=(10, 10, 100, 40), name="Btn 1"):
    monkeypatch.setattr(session_module, "get_ui_tree", MagicMock(return_value=[{"name": name}]))
    monkeypatch.setattr(
        session_module,
        "find_element",
        MagicMock(return_value={"bounding_box": bounding_box, "name": name, "confidence": 1.0, "match_method": "exact"}),
    )
    monkeypatch.setattr(session_module, "render_guidance", MagicMock(return_value="/tmp/fake_screenshot.png"))


def mock_grounding_failure(monkeypatch):
    monkeypatch.setattr(session_module, "get_ui_tree", MagicMock(return_value=[]))
    monkeypatch.setattr(session_module, "find_element", MagicMock(return_value=None))
    monkeypatch.setattr(session_module, "render_guidance", MagicMock())


# --- ask(): starting a new question -----------------------------------------


def test_ask_with_no_workflow_starts_procedural_workflow(monkeypatch):
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_procedural_routed()))
    mock_grounding_success(monkeypatch)

    session = GuidanceSession("OpenShot")
    response = session.ask("how do I export a video")

    assert response["type"] == "step"
    assert response["goal"] == "Export your project to a video file"
    assert response["instruction"] == "Step 1 instruction"
    assert response["image_path"] == "/tmp/fake_screenshot.png"
    assert session.workflow is not None
    assert session.current_step_index == 0


def test_ask_with_no_workflow_reference_intent_returns_answer_no_workflow(monkeypatch):
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_reference_routed()))

    session = GuidanceSession("OpenShot")
    response = session.ask("what does resolution mean")

    assert response == {"type": "answer", "message": "Resolution: output size"}
    assert session.workflow is None


def test_ask_with_no_workflow_unknown_intent_returns_no_match(monkeypatch):
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_unknown_routed()))

    session = GuidanceSession("OpenShot")
    response = session.ask("asdkjaslkdj nonsense")

    assert response["type"] == "no_match"
    assert session.workflow is None


# --- ask(): continuing an active workflow -----------------------------------


def test_continue_workflow_rule_based_advance_does_not_call_llm(monkeypatch):
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_procedural_routed(num_steps=2)))
    mock_grounding_success(monkeypatch)
    mock_generate_json = MagicMock()
    monkeypatch.setattr(session_module, "generate_json", mock_generate_json)

    session = GuidanceSession("OpenShot")
    session.ask("how do I export a video")

    response = session.ask("done")

    assert response["classified_via"] == "rule"
    assert response["instruction"] == "Step 2 instruction"
    assert session.current_step_index == 1
    mock_generate_json.assert_not_called()


def test_continue_workflow_advance_past_last_step_completes(monkeypatch):
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_procedural_routed(num_steps=1)))
    mock_grounding_success(monkeypatch)

    session = GuidanceSession("OpenShot")
    session.ask("how do I export a video")

    response = session.ask("done")

    assert response["type"] == "complete"
    assert session.workflow is None
    assert session.current_step_index == 0


def test_continue_workflow_rule_based_stuck_stays_on_same_step(monkeypatch):
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_procedural_routed(num_steps=2)))
    mock_grounding_success(monkeypatch)
    mock_generate_json = MagicMock()
    monkeypatch.setattr(session_module, "generate_json", mock_generate_json)

    session = GuidanceSession("OpenShot")
    session.ask("how do I export a video")

    response = session.ask("no, didn't work")

    assert response["classified_via"] == "rule"
    assert response["instruction"] == "Step 1 instruction"
    assert response["note_prefix"] == "No worries -- here's that step again:"
    assert session.current_step_index == 0
    mock_generate_json.assert_not_called()


def test_continue_workflow_ambiguous_reply_falls_back_to_llm(monkeypatch):
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_procedural_routed(num_steps=2)))
    mock_grounding_success(monkeypatch)
    monkeypatch.setattr(
        session_module, "generate_json", MagicMock(return_value=ReplyClassification(intent="advance"))
    )

    session = GuidanceSession("OpenShot")
    session.ask("how do I export a video")

    response = session.ask("hmm i think so? not totally sure")

    assert response["classified_via"] == "llm"
    assert session.current_step_index == 1
    session_module.generate_json.assert_called_once()


def test_continue_workflow_llm_classifies_new_question_resets_workflow(monkeypatch):
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_procedural_routed(num_steps=2)))
    mock_grounding_success(monkeypatch)

    session = GuidanceSession("OpenShot")
    session.ask("how do I export a video")

    # Ambiguous reply -> LLM says "new_question" -> route() gets called
    # again with the new message and returns a reference answer instead.
    monkeypatch.setattr(
        session_module, "generate_json", MagicMock(return_value=ReplyClassification(intent="new_question"))
    )
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_reference_routed()))

    response = session.ask("actually what does resolution mean")

    assert response == {"type": "answer", "message": "Resolution: output size"}
    assert session.workflow is None


def test_llm_classify_reply_defaults_to_stuck_on_generation_error(monkeypatch):
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_procedural_routed(num_steps=2)))
    mock_grounding_success(monkeypatch)
    monkeypatch.setattr(session_module, "generate_json", MagicMock(side_effect=GenerationError("model gave up")))

    session = GuidanceSession("OpenShot")
    session.ask("how do I export a video")

    response = session.ask("hmm i think so? not totally sure")

    assert response["classified_via"] == "llm"
    assert session.current_step_index == 0  # stayed put, treated as "stuck"
    assert response["note_prefix"] == "No worries -- here's that step again:"


# --- present_current_step() -------------------------------------------------


def test_present_current_step_resolved_includes_image_path_and_grounding(monkeypatch):
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_procedural_routed(grounding="canvas")))
    mock_grounding_success(monkeypatch, bounding_box=(1, 2, 3, 4), name="Btn 1")

    session = GuidanceSession("OpenShot")
    response = session.ask("how do I export a video")

    assert response["image_path"] == "/tmp/fake_screenshot.png"
    assert response["grounding_confidence"] == "canvas"
    session_module.render_guidance.assert_called_once_with("OpenShot", (1, 2, 3, 4), "Btn 1")


def test_present_current_step_unresolved_returns_none_image_and_note(monkeypatch):
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=make_procedural_routed()))
    mock_grounding_failure(monkeypatch)

    session = GuidanceSession("OpenShot")
    response = session.ask("how do I export a video")

    assert response["image_path"] is None
    assert response["note"] == "couldn't pinpoint this on screen, here's what to do in words"
    session_module.render_guidance.assert_not_called()


def test_present_current_step_raises_without_active_workflow():
    session = GuidanceSession("OpenShot")

    with pytest.raises(RuntimeError):
        session.present_current_step()


def test_present_current_step_passes_glossary_context_to_matcher(monkeypatch):
    routed = make_procedural_routed()
    monkeypatch.setattr(session_module, "route", MagicMock(return_value=routed))
    mock_grounding_success(monkeypatch)

    session = GuidanceSession("OpenShot")
    session.ask("how do I export a video")

    _, kwargs = session_module.find_element.call_args
    assert kwargs["glossary_context"] == routed["glossary_context"]


# --- _rule_based_classify() pure logic --------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("done", "advance"),
        ("Yes!", "advance"),
        ("ok next", "advance"),
        ("got it, thanks", "advance"),
        ("stuck", "stuck"),
        ("no", "stuck"),
        ("that didn't work", "stuck"),
        ("I'm confused", "stuck"),
    ],
)
def test_rule_based_classify_clear_cases(message, expected):
    assert _rule_based_classify(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "I know how to do this already",  # contains "no" inside "know" -- must not false-positive as "stuck"
        "hmm not totally sure",
        "what does resolution mean",
        "",
    ],
)
def test_rule_based_classify_ambiguous_returns_none(message):
    assert _rule_based_classify(message) is None
