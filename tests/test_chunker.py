from ingestion.chunker import split_into_chunks

SAMPLE_TEXT = """Some preamble text before the first heading, should be discarded.

1.6.7 Properties
This is the body text of the Properties section.
It spans multiple lines.

1.6.8 Advanced Properties
This is the next section's body.
"""


def test_split_into_chunks_detects_numbered_headings():
    chunks = split_into_chunks(SAMPLE_TEXT)

    assert len(chunks) == 2
    assert chunks[0].section_number == "1.6.7"
    assert chunks[0].title == "Properties"
    assert "body text of the Properties section" in chunks[0].body_text
    assert chunks[1].section_number == "1.6.8"
    assert chunks[1].title == "Advanced Properties"


def test_split_into_chunks_body_excludes_next_heading():
    chunks = split_into_chunks(SAMPLE_TEXT)

    assert "1.6.8" not in chunks[0].body_text


def test_split_into_chunks_configurable_pattern():
    text = "Section A: Intro\nSome body text.\n\nSection B: Next\nMore text.\n"
    pattern = r"^Section (?P<number>[A-Z]): (?P<title>[^\n]+)$"

    chunks = split_into_chunks(text, heading_pattern=pattern)

    assert len(chunks) == 2
    assert chunks[0].section_number == "A"
    assert chunks[0].title == "Intro"
    assert chunks[1].section_number == "B"
    assert chunks[1].title == "Next"


def test_split_into_chunks_no_matches_returns_empty():
    assert split_into_chunks("No headings here at all.") == []
