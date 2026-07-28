from perception.matcher import find_element


def make_tree():
    return [
        {"name": "Export Video", "bounding_box": (10, 10, 110, 40), "control_type": "ButtonControl"},
        {"name": "Timeline", "bounding_box": (0, 200, 800, 400), "control_type": "PaneControl"},
        {"name": "Export Project", "bounding_box": (5, 5, 200, 25), "control_type": "MenuItemControl"},
    ]


def test_find_element_exact_match_button():
    hint = {"type": "button", "label": "Export Video"}

    result = find_element(hint, make_tree())

    assert result is not None
    assert result["name"] == "Export Video"
    assert result["bounding_box"] == (10, 10, 110, 40)
    assert result["confidence"] == 1.0
    assert result["match_method"] == "exact"


def test_find_element_exact_match_is_case_insensitive():
    hint = {"type": "button", "label": "export video"}

    result = find_element(hint, make_tree())

    assert result is not None
    assert result["name"] == "Export Video"


def test_find_element_menu_hint_matches_last_path_segment():
    hint = {"type": "menu", "path": ["File", "Export Project"]}

    result = find_element(hint, make_tree())

    assert result is not None
    assert result["name"] == "Export Project"
    assert result["match_method"] == "exact"


def test_find_element_fuzzy_match_above_threshold():
    hint = {"type": "button", "label": "Export Vidoe"}  # typo

    result = find_element(hint, make_tree())

    assert result is not None
    assert result["name"] == "Export Video"
    assert result["match_method"] == "fuzzy"
    assert 0.0 < result["confidence"] < 1.0


def test_find_element_returns_none_when_nothing_clears_threshold():
    hint = {"type": "button", "label": "Completely Unrelated Widget Name"}

    result = find_element(hint, make_tree())

    assert result is None


def test_find_element_returns_none_for_empty_hint():
    assert find_element({"type": "unknown"}, make_tree()) is None
    assert find_element({"type": "menu", "path": []}, make_tree()) is None


def test_find_element_glossary_assisted_retry_resolves_indirect_reference():
    # "video export" is a real-world case where the workflow's phrasing
    # doesn't score high enough directly against the tree's raw control
    # name ("Export Video File"), but a glossary entry's canonical
    # element_name ("Export Video") bridges the gap: the hint matches the
    # glossary entry well, and the glossary entry's name in turn fuzzy-
    # matches the tree. (Scores verified empirically with rapidfuzz before
    # writing this test — see the session notes.)
    tree = [
        {"name": "Export Video File", "bounding_box": (10, 10, 110, 40), "control_type": "ButtonControl"},
        {"name": "Timeline", "bounding_box": (0, 200, 800, 400), "control_type": "PaneControl"},
    ]
    hint = {"type": "unknown", "label": "video export"}
    glossary_context = [
        {"element_name": "Export Video", "context": "Main toolbar", "description": "Exports the current project."},
    ]

    # Confirm the premise: a direct match (no glossary) fails.
    assert find_element(hint, tree) is None

    result = find_element(hint, tree, glossary_context=glossary_context)

    assert result is not None
    assert result["name"] == "Export Video File"
    assert result["match_method"] in ("glossary_exact", "glossary_fuzzy")
    assert result["confidence"] < 1.0  # glossary-indirected matches are discounted


def test_find_element_glossary_context_not_used_when_direct_match_succeeds():
    hint = {"type": "button", "label": "Export Video"}
    glossary_context = [{"element_name": "Timeline", "context": "Main window", "description": "..."}]

    result = find_element(hint, make_tree(), glossary_context=glossary_context)

    assert result["match_method"] == "exact"


def test_find_element_returns_none_when_glossary_also_fails():
    hint = {"type": "unknown", "label": "Completely Unrelated Widget Name"}
    glossary_context = [{"element_name": "Timeline", "context": "Main window", "description": "..."}]

    result = find_element(hint, make_tree(), glossary_context=glossary_context)

    assert result is None
