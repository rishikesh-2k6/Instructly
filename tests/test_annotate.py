from PIL import Image

from guidance.annotate import annotate_target


def blank_image(size=(400, 300), color=(50, 50, 50)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_annotate_target_returns_same_size_image():
    image = blank_image()
    result = annotate_target(image, (100, 100, 200, 150))

    assert result.size == image.size


def test_annotate_target_does_not_mutate_input():
    image = blank_image()
    original_pixel = image.getpixel((150, 100))

    annotate_target(image, (100, 80, 200, 120))

    assert image.getpixel((150, 100)) == original_pixel


def test_annotate_target_draws_a_visible_highlight():
    image = blank_image(color=(50, 50, 50))
    box = (100, 100, 200, 150)

    result = annotate_target(image, box)

    # A pixel on the box's border should no longer be the flat background color.
    border_pixel = result.getpixel((100, 125))
    assert border_pixel != (50, 50, 50)


def test_annotate_target_without_label_draws_no_callout_text_area():
    image = blank_image()
    box = (100, 100, 200, 150)

    without_label = annotate_target(image, box)
    with_label = annotate_target(image, box, label="Export Video")

    # Region above the box (where a callout would go) should differ once a
    # label is added, but not otherwise.
    region_without = without_label.crop((80, 60, 220, 95))
    region_with = with_label.crop((80, 60, 220, 95))
    assert list(region_without.getdata()) != list(region_with.getdata())


def test_annotate_target_callout_stays_within_bounds_near_top_left_corner():
    image = blank_image(size=(400, 300))
    # A box right at the top-left corner forces the callout's default
    # above-the-box placement to be invalid on both axes.
    box = (0, 0, 40, 20)

    result = annotate_target(image, box, label="Tiny Corner Button With A Long Label")

    # No non-background pixel should appear outside the image (trivially
    # true since PIL clips draws to the canvas) — assert instead that the
    # callout's background color appears within EDGE_MARGIN of an edge,
    # not that it was cut off. We check indirectly: the top-left EDGE_MARGIN
    # strip should contain the callout background color (proving it was
    # clamped inward, not drawn off-canvas and lost).
    from guidance.annotate import CALLOUT_BG

    found_callout_pixel = False
    for x in range(0, min(300, image.width)):
        for y in range(0, min(80, image.height)):
            r, g, b = result.getpixel((x, y))
            if abs(r - CALLOUT_BG[0]) < 10 and abs(g - CALLOUT_BG[1]) < 10 and abs(b - CALLOUT_BG[2]) < 10:
                found_callout_pixel = True
                break
        if found_callout_pixel:
            break

    assert found_callout_pixel


def test_annotate_target_callout_never_renders_past_image_edges():
    image = blank_image(size=(400, 300))
    box = (390, 290, 400, 300)  # bottom-right corner

    # Should not raise, and should complete without drawing outside bounds
    # (PIL itself clips, but we verify no exception from negative/overflow
    # geometry in the clamping math).
    result = annotate_target(image, box, label="Bottom Right")

    assert result.size == image.size
