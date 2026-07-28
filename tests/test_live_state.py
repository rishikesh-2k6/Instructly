from unittest.mock import MagicMock

from perception import live_state


def test_get_active_window_ui_tree_returns_none_when_nothing_focused(monkeypatch):
    monkeypatch.setattr(live_state.auto, "GetFocusedControl", MagicMock(return_value=None))

    assert live_state.get_active_window_ui_tree() is None


def test_get_active_window_ui_tree_returns_none_when_top_level_unresolvable(monkeypatch):
    focused = MagicMock()
    focused.GetTopLevelControl.return_value = None
    monkeypatch.setattr(live_state.auto, "GetFocusedControl", MagicMock(return_value=focused))

    assert live_state.get_active_window_ui_tree() is None


def test_get_active_window_ui_tree_returns_none_when_title_empty(monkeypatch):
    top_level = MagicMock()
    top_level.Name = ""
    focused = MagicMock()
    focused.GetTopLevelControl.return_value = top_level
    monkeypatch.setattr(live_state.auto, "GetFocusedControl", MagicMock(return_value=focused))

    assert live_state.get_active_window_ui_tree() is None


def test_get_active_window_ui_tree_returns_title_and_tree(monkeypatch):
    top_level = MagicMock()
    top_level.Name = "OpenShot Video Editor"
    focused = MagicMock()
    focused.GetTopLevelControl.return_value = top_level
    monkeypatch.setattr(live_state.auto, "GetFocusedControl", MagicMock(return_value=focused))

    fake_nodes = [{"name": "OpenShot Video Editor", "depth": 0}]
    mock_walk = MagicMock(return_value=fake_nodes)
    monkeypatch.setattr(live_state, "_walk", mock_walk)

    result = live_state.get_active_window_ui_tree(max_depth=3)

    assert result == ("OpenShot Video Editor", fake_nodes)
    mock_walk.assert_called_once_with(top_level, 3)


def test_get_active_window_ui_tree_returns_none_on_any_com_error(monkeypatch):
    def raising_get_focused_control():
        raise OSError("COM call failed")

    monkeypatch.setattr(live_state.auto, "GetFocusedControl", raising_get_focused_control)

    assert live_state.get_active_window_ui_tree() is None
