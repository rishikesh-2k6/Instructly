"""Chat orchestration: ties retrieval, perception, and guidance together
into a single conversational loop.

Design note on the reply classifier: after a workflow starts, every user
reply first goes through a cheap keyword classifier (_rule_based_classify).
Only when that's ambiguous does a reply go to the local Ollama model via
ingestion/ollama_client.py's generate_json. Local inference is slow enough
that classifying every turn with it would make step transitions feel
laggy, so the rule-based path is the common case, not a fallback of last
resort. See chat/manual_test.md for how this feels in practice.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel

from guidance.render import render_guidance
from ingestion.ollama_client import GenerationError, generate_json
from perception.matcher import find_element
from perception.uia_tree import get_ui_tree
from retrieval.router import route

# --- rule-based reply classification ---------------------------------------

ADVANCE_WORDS = {"done", "next", "ok", "okay", "yes", "yep", "yeah", "finished", "complete", "worked", "continue"}
STUCK_WORDS = {"stuck", "help", "no", "nope", "cant", "confused", "lost"}

ADVANCE_PHRASES = ("got it", "did it", "moving on", "that worked")
STUCK_PHRASES = ("not working", "didnt work", "doesnt work", "where is", "wheres", "cant find", "don't see", "dont see")


def _normalize(text: str) -> str:
    return text.strip().lower().replace("'", "")


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", _normalize(text)))


def _rule_based_classify(message: str) -> Optional[Literal["advance", "stuck"]]:
    """Cheap keyword classification of a reply to "did that work?".

    Returns "advance", "stuck", or None if the message doesn't clearly
    match exactly one bucket — ambiguity is the caller's signal to fall
    back to an Ollama call rather than guess.
    """
    normalized = _normalize(message)
    tokens = _tokenize(message)

    matched = set()
    if tokens & ADVANCE_WORDS or any(phrase in normalized for phrase in ADVANCE_PHRASES):
        matched.add("advance")
    if tokens & STUCK_WORDS or any(phrase in normalized for phrase in STUCK_PHRASES):
        matched.add("stuck")

    if len(matched) == 1:
        return matched.pop()
    return None


class ReplyClassification(BaseModel):
    intent: Literal["advance", "stuck", "new_question"]


REPLY_CLASSIFICATION_PROMPT = """The user is following a step-by-step guide. The current step is:

"{step_instruction}"

Their reply was: "{message}"

Classify their reply into exactly one category:
- "advance": they completed the step and are ready to move on.
- "stuck": they're having trouble with this step and want more help with it.
- "new_question": their reply isn't about this step at all -- it's a new, unrelated question.

Return ONLY JSON: {{"intent": "<one of advance, stuck, new_question>"}}"""


def _llm_classify_reply(message: str, step_instruction: str) -> Literal["advance", "stuck", "new_question"]:
    prompt = REPLY_CLASSIFICATION_PROMPT.format(step_instruction=step_instruction, message=message)
    try:
        result = generate_json(prompt, ReplyClassification)
        return result.intent
    except GenerationError:
        # If even the LLM can't produce a clean classification, "stuck" is
        # the safest default: it re-offers help on the current step rather
        # than silently advancing past a step the user might not have
        # actually finished, or dropping their message as a new question.
        return "stuck"


# --- session ------------------------------------------------------------


class GuidanceSession:
    """Holds the conversational state for guiding one user through one app.

    current_step_index advances as the user confirms steps are done, and
    resets to 0 whenever a new workflow starts (a fresh procedural
    question replaces whatever workflow, if any, was previously active).
    """

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.workflow: Optional[dict] = None  # a routed retrieval result: {"intent", "primary", "glossary_context"}
        self.current_step_index: int = 0
        self.history: list[dict] = []

    @property
    def _steps(self) -> list[dict]:
        if not self.workflow:
            return []
        return self.workflow["primary"].get("steps", [])

    def _record(self, user_message: str, response: dict) -> dict:
        self.history.append({"role": "user", "message": user_message})
        self.history.append({"role": "assistant", "response": response})
        return response

    def _start_new_question(self, user_message: str) -> dict:
        routed = route(user_message, self.app_name)
        intent = routed["intent"]
        primary = routed["primary"]

        if intent == "procedural" and primary and primary.get("steps"):
            self.workflow = routed
            self.current_step_index = 0
            response = {"type": "step", "goal": primary.get("goal"), **self.present_current_step()}
        elif intent == "reference":
            self.workflow = None
            response = {
                "type": "answer",
                "message": primary["snippet"] if primary else "I couldn't find anything about that.",
            }
        else:
            self.workflow = None
            response = {
                "type": "no_match",
                "message": "I couldn't find a clear how-to or reference answer for that in the ingested docs.",
            }

        return self._record(user_message, response)

    def _continue_workflow(self, user_message: str) -> dict:
        step_instruction = self._steps[self.current_step_index]["instruction"]

        reply_intent = _rule_based_classify(user_message)
        classified_via = "rule"
        if reply_intent is None:
            reply_intent = _llm_classify_reply(user_message, step_instruction)
            classified_via = "llm"

        if reply_intent == "new_question":
            # _start_new_question already records history; return directly.
            return self._start_new_question(user_message)

        if reply_intent == "advance":
            self.current_step_index += 1
            if self.current_step_index >= len(self._steps):
                self.workflow = None
                self.current_step_index = 0
                response = {"type": "complete", "message": "That's the last step -- you're done!"}
            else:
                response = {"type": "step", **self.present_current_step()}
        else:  # "stuck"
            response = {
                "type": "step",
                "note_prefix": "No worries -- here's that step again:",
                **self.present_current_step(),
            }

        response["classified_via"] = classified_via
        return self._record(user_message, response)

    def ask(self, user_message: str) -> dict:
        """Handle one turn of the conversation and return a response dict.

        With no active workflow, a message is treated as a new question:
        a "procedural" retrieval intent starts walking that workflow from
        step 0; a "reference" intent is answered directly, with no step-
        walking. With an active workflow, a message is treated as a reply
        to "did that work?" -- see _continue_workflow.
        """
        if self.workflow is None:
            return self._start_new_question(user_message)
        return self._continue_workflow(user_message)

    def present_current_step(self) -> dict:
        """Resolve the current step onto the live screen and render guidance
        for it.

        Always returns something useful even when grounding fails --
        image_path is None with a "note" explaining why, never a silent
        failure. Uses uia_tree.get_ui_tree(self.app_name) rather than
        live_state.get_active_window_ui_tree(): the target app is already
        known here (it's the workflow's app_name), and grabbing whatever
        window currently has OS focus instead could easily be the wrong
        window -- e.g. the terminal this chat loop is running in.
        """
        if not self.workflow:
            raise RuntimeError("present_current_step() called with no active workflow")

        step = self._steps[self.current_step_index]
        instruction = step["instruction"]
        ui_hint = step.get("ui_hint") or {}

        ui_tree = get_ui_tree(self.app_name)
        match = find_element(ui_hint, ui_tree, glossary_context=self.workflow["glossary_context"])

        if match is None:
            return {
                "instruction": instruction,
                "image_path": None,
                "note": "couldn't pinpoint this on screen, here's what to do in words",
            }

        image_path = render_guidance(self.app_name, match["bounding_box"], match["name"])
        return {
            "instruction": instruction,
            "image_path": image_path,
            "grounding_confidence": self.workflow["primary"].get("grounding_confidence"),
        }
