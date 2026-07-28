"""Simple command-line chat loop for GuidanceSession.

Not a polished UI -- proves the retrieval -> perception -> guidance loop
works end to end against a real running app and real ingested docs, with
local Ollama doing the thinking. A future session builds a real GUI on top
of chat/session.py instead of this CLI.

Usage:
    python -m chat.cli --app-name OpenShot
"""

from __future__ import annotations

import argparse

from chat.session import GuidanceSession


def _safe(text: str) -> str:
    # Some real window/control names hit characters this console's
    # encoding can't render (see perception/manual_test.py) -- degrade to
    # escaped ASCII rather than crash the loop mid-conversation.
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _print_response(response: dict) -> None:
    response_type = response.get("type")

    if response_type in ("answer", "no_match", "complete"):
        print(_safe(response["message"]))
        return

    if response_type == "step":
        if response.get("goal"):
            print(f"Goal: {_safe(response['goal'])}")
        if response.get("note_prefix"):
            print(_safe(response["note_prefix"]))
        print(f"-> {_safe(response['instruction'])}")
        if response.get("image_path"):
            print(f"   [screenshot: {response['image_path']}]")
        elif response.get("note"):
            print(f"   ({_safe(response['note'])})")
        if response.get("grounding_confidence"):
            print(f"   grounding: {response['grounding_confidence']}")
        return

    print(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Command-line chat loop for Instructly's guidance session.")
    parser.add_argument("--app-name", required=True, help='Target app name, e.g. "OpenShot".')
    args = parser.parse_args()

    session = GuidanceSession(args.app_name)
    print(f"Instructly -- ask a question about {args.app_name}. Ctrl+C to quit.\n")

    while True:
        try:
            user_message = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_message:
            continue

        response = session.ask(user_message)
        _print_response(response)
        print()


if __name__ == "__main__":
    main()
