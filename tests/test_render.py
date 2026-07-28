import os
from unittest.mock import MagicMock

from PIL import Image

from guidance import render


def test_render_guidance_converts_screen_absolute_box_to_image_relative(monkeypatch, tmp_path):
    fake_image = Image.new("RGB", (800, 600), (10, 10, 10))
    monkeypatch.setattr(render, "capture_window_screenshot", MagicMock(return_value=fake_image))
    monkeypatch.setattr(render, "get_window_bounds", MagicMock(return_value=(100, 200, 900, 800)))

    mock_annotate = MagicMock(return_value=fake_image)
    monkeypatch.setattr(render, "annotate_target", mock_annotate)

    path = render.render_guidance("OpenShot", (150, 250, 250, 300), "Export Video")

    # window origin is (100, 200) -> screen box (150,250,250,300) becomes
    # image-relative (50, 50, 150, 100)
    mock_annotate.assert_called_once_with(fake_image, (50, 50, 150, 100), "Export Video")
    assert os.path.exists(path)
    os.remove(path)


def test_render_guidance_uses_zero_origin_when_window_not_found(monkeypatch):
    fake_image = Image.new("RGB", (800, 600), (10, 10, 10))
    monkeypatch.setattr(render, "capture_window_screenshot", MagicMock(return_value=fake_image))
    monkeypatch.setattr(render, "get_window_bounds", MagicMock(return_value=None))

    mock_annotate = MagicMock(return_value=fake_image)
    monkeypatch.setattr(render, "annotate_target", mock_annotate)

    path = render.render_guidance("OpenShot", (150, 250, 250, 300), "Export Video")

    mock_annotate.assert_called_once_with(fake_image, (150, 250, 250, 300), "Export Video")
    os.remove(path)


def test_render_guidance_saves_and_returns_a_png_path(monkeypatch):
    fake_image = Image.new("RGB", (100, 100), (0, 0, 0))
    monkeypatch.setattr(render, "capture_window_screenshot", MagicMock(return_value=fake_image))
    monkeypatch.setattr(render, "get_window_bounds", MagicMock(return_value=None))
    monkeypatch.setattr(render, "annotate_target", MagicMock(return_value=fake_image))

    path = render.render_guidance("OpenShot", (10, 10, 20, 20), "Some Button")

    assert path.endswith(".png")
    assert os.path.exists(path)
    with Image.open(path) as saved:
        assert saved.size == (100, 100)
    os.remove(path)
