"""Manual visual test for perception. Not a pytest test — it needs a real
GUI on screen. Run directly:

    python -m perception.manual_test
"""

from __future__ import annotations

from perception.matcher import find_element
from perception.uia_tree import get_ui_tree

WINDOW_TITLE = "OpenShot"

EXAMPLE_HINTS = [
    {"type": "button", "label": "Export Video"},
    {"type": "menu", "path": ["File", "Export Project"]},
    {"type": "unknown", "label": "Some Nonexistent Button XYZ"},
]


def _safe(text: str) -> str:
    # Some real window/control names contain characters this console's
    # cp1252 encoding can't render (observed directly while building this
    # module) — degrade to escaped ASCII rather than crash the script.
    return text.encode("ascii", "backslashreplace").decode("ascii")


def main() -> None:
    input(f"Open {WINDOW_TITLE} and make sure it's visible on screen, then press Enter...")

    print("Walking UI tree...")
    tree = get_ui_tree(WINDOW_TITLE)
    print(f"Found {len(tree)} controls.\n")

    for hint in EXAMPLE_HINTS:
        result = find_element(hint, tree)
        print(f"hint: {hint}")
        if result is None:
            print("  -> NOT FOUND")
        else:
            print(
                f"  -> {_safe(result['name'])!r} at {result['bounding_box']} "
                f"(confidence={result['confidence']:.2f}, method={result['match_method']})"
            )
        print()


if __name__ == "__main__":
    main()
