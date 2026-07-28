from types import SimpleNamespace
from unittest.mock import MagicMock

from perception import uia_tree


def make_control(name, control_type="ButtonControl", automation_id="", class_name="", box=(0, 0, 10, 10),
                  enabled=True, children=None):
    rect = SimpleNamespace(left=box[0], top=box[1], right=box[2], bottom=box[3])
    control = MagicMock()
    control.Name = name
    control.ControlTypeName = control_type
    control.AutomationId = automation_id
    control.ClassName = class_name
    control.BoundingRectangle = rect
    control.IsEnabled = enabled
    control.GetChildren.return_value = children or []
    return control


def test_walk_flattens_tree_with_correct_depth():
    grandchild = make_control("Grandchild", box=(1, 1, 2, 2))
    child = make_control("Child", box=(0, 0, 5, 5), children=[grandchild])
    root = make_control("Root", box=(0, 0, 100, 100), children=[child])

    nodes = uia_tree._walk(root, max_depth=6)

    assert [n["name"] for n in nodes] == ["Root", "Child", "Grandchild"]
    assert [n["depth"] for n in nodes] == [0, 1, 2]


def test_walk_respects_max_depth():
    grandchild = make_control("Grandchild")
    child = make_control("Child", children=[grandchild])
    root = make_control("Root", children=[child])

    nodes = uia_tree._walk(root, max_depth=1)

    assert [n["name"] for n in nodes] == ["Root", "Child"]


def test_walk_node_shape():
    root = make_control(
        "Export Video",
        control_type="ButtonControl",
        automation_id="exportBtn",
        class_name="QPushButton",
        box=(10, 20, 110, 50),
        enabled=True,
    )

    nodes = uia_tree._walk(root, max_depth=6)

    assert nodes[0] == {
        "control_type": "ButtonControl",
        "name": "Export Video",
        "automation_id": "exportBtn",
        "class_name": "QPushButton",
        "bounding_box": (10, 20, 110, 50),
        "is_enabled": True,
        "depth": 0,
    }


def test_walk_skips_control_that_raises_on_property_access():
    good_child = make_control("Good Child")
    bad_child = MagicMock()
    type(bad_child).Name = property(lambda self: (_ for _ in ()).throw(RuntimeError("stale element")))
    root = make_control("Root", children=[bad_child, good_child])

    nodes = uia_tree._walk(root, max_depth=6)

    assert [n["name"] for n in nodes] == ["Root", "Good Child"]


def test_walk_skips_subtree_when_get_children_raises():
    child = make_control("Child")
    child.GetChildren.side_effect = RuntimeError("element disappeared")
    root = make_control("Root", children=[child])

    nodes = uia_tree._walk(root, max_depth=6)

    assert [n["name"] for n in nodes] == ["Root", "Child"]


def test_find_top_level_window_matches_case_insensitive_substring(monkeypatch):
    match = make_control("OpenShot Video Editor - untitled")
    other = make_control("Notepad")
    root = make_control("Desktop", children=[other, match])
    monkeypatch.setattr(uia_tree.auto, "GetRootControl", MagicMock(return_value=root))

    found = uia_tree._find_top_level_window("openshot")

    assert found is match


def test_find_top_level_window_returns_none_when_no_match(monkeypatch):
    root = make_control("Desktop", children=[make_control("Notepad")])
    monkeypatch.setattr(uia_tree.auto, "GetRootControl", MagicMock(return_value=root))

    assert uia_tree._find_top_level_window("nonexistent app") is None


def test_get_ui_tree_returns_empty_list_when_window_not_found(monkeypatch):
    monkeypatch.setattr(uia_tree, "_find_top_level_window", MagicMock(return_value=None))

    assert uia_tree.get_ui_tree("nonexistent app") == []


def test_get_ui_tree_walks_matched_window(monkeypatch):
    window = make_control("OpenShot", children=[make_control("Export Button")])
    monkeypatch.setattr(uia_tree, "_find_top_level_window", MagicMock(return_value=window))

    nodes = uia_tree.get_ui_tree("OpenShot")

    assert [n["name"] for n in nodes] == ["OpenShot", "Export Button"]
