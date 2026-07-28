"""Manual visual test for guidance rendering. Not a pytest test — it needs
a real GUI on screen. Run directly:

    python -m guidance.manual_test
"""

from __future__ import annotations

import os

from guidance.render import render_guidance

WINDOW_TITLE = "OpenShot"

# NOTE: perception/ hasn't been built yet in this project (that session is
# still blocked on missing audit_openshot.py / audit_output.json input), so
# this is NOT a real box pulled from perception/manual_test.py's output as
# originally intended — it's a rough placeholder guess at where OpenShot's
# "Export Video" toolbar button typically sits on a 1920x1080 screen.
# Replace this with a real find_element() result once perception/ exists.
EXAMPLE_BOUNDING_BOX = (1100, 80, 1180, 110)
EXAMPLE_LABEL = "Export Video"


def main() -> None:
    input(f"Open {WINDOW_TITLE} and make sure it's visible on screen, then press Enter...")

    path = render_guidance(WINDOW_TITLE, EXAMPLE_BOUNDING_BOX, EXAMPLE_LABEL)
    print(f"Saved annotated screenshot to: {path}")

    os.startfile(path)


if __name__ == "__main__":
    main()
