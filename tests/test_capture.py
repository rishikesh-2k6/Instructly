import warnings
from unittest.mock import MagicMock

from guidance import capture


def test_find_window_matches_case_insensitive_substring(monkeypatch):
    def fake_enum_windows(callback, extra):
        callback(1, extra)  # not visible, should be skipped
        callback(2, extra)  # visible but no match
        callback(3, extra)  # visible and matches

    visibility = {1: False, 2: True, 3: True}
    titles = {1: "OpenShot Video Editor", 2: "Notepad", 3: "OpenShot Video Editor - untitled"}

    monkeypatch.setattr(capture.win32gui, "EnumWindows", fake_enum_windows)
    monkeypatch.setattr(capture.win32gui, "IsWindowVisible", lambda hwnd: visibility[hwnd])
    monkeypatch.setattr(capture.win32gui, "GetWindowText", lambda hwnd: titles[hwnd])

    hwnd = capture._find_window("openshot")

    assert hwnd == 3


def test_find_window_returns_none_when_no_match(monkeypatch):
    monkeypatch.setattr(capture.win32gui, "EnumWindows", lambda callback, extra: None)

    assert capture._find_window("nonexistent app") is None


def test_get_window_bounds_returns_none_when_window_not_found(monkeypatch):
    monkeypatch.setattr(capture, "_find_window", MagicMock(return_value=None))

    assert capture.get_window_bounds("nonexistent app") is None


def test_get_window_bounds_uses_dwm_extended_frame_when_available(monkeypatch):
    monkeypatch.setattr(capture, "_find_window", MagicMock(return_value=42))

    class FakeRect:
        left, top, right, bottom = 10, 20, 810, 620

    def fake_dwm_get_window_attribute(hwnd, attribute, rect_ref, size):
        rect_ref._obj.left = 10
        rect_ref._obj.top = 20
        rect_ref._obj.right = 810
        rect_ref._obj.bottom = 620
        return 0  # S_OK

    monkeypatch.setattr(capture.ctypes.windll.dwmapi, "DwmGetWindowAttribute", fake_dwm_get_window_attribute)

    bounds = capture.get_window_bounds("OpenShot")

    assert bounds == (10, 20, 810, 620)


def test_get_window_bounds_falls_back_to_get_window_rect_on_dwm_failure(monkeypatch):
    monkeypatch.setattr(capture, "_find_window", MagicMock(return_value=42))

    def raising_dwm_call(*args, **kwargs):
        raise OSError("dwmapi not available")

    monkeypatch.setattr(capture.ctypes.windll.dwmapi, "DwmGetWindowAttribute", raising_dwm_call)
    monkeypatch.setattr(capture.win32gui, "GetWindowRect", lambda hwnd: (5, 5, 400, 300))

    bounds = capture.get_window_bounds("OpenShot")

    assert bounds == (5, 5, 400, 300)


def test_capture_window_screenshot_uses_matched_window_bounds(monkeypatch):
    monkeypatch.setattr(capture, "get_window_bounds", MagicMock(return_value=(10, 20, 810, 620)))
    mock_grab = MagicMock(return_value="fake-image")
    monkeypatch.setattr(capture.ImageGrab, "grab", mock_grab)

    result = capture.capture_window_screenshot("OpenShot")

    assert result == "fake-image"
    mock_grab.assert_called_once_with(bbox=(10, 20, 810, 620))


def test_capture_window_screenshot_falls_back_to_fullscreen_with_warning(monkeypatch):
    monkeypatch.setattr(capture, "get_window_bounds", MagicMock(return_value=None))
    mock_grab = MagicMock(return_value="fake-fullscreen-image")
    monkeypatch.setattr(capture.ImageGrab, "grab", mock_grab)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = capture.capture_window_screenshot("Nonexistent App")

    assert result == "fake-fullscreen-image"
    mock_grab.assert_called_once_with()
    assert any("falling back to a full-screen capture" in str(w.message) for w in caught)
