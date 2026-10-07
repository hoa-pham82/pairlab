"""Tests for Markdown chunking."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from ml.rag.chunking import chunk_directory, chunk_markdown

DOC = """# Title

Intro words here.

## What

alpha beta gamma

## How

one two three four five six seven eight nine ten
"""


def test_sections_become_chunks_with_headings_and_ids():
    chunks = chunk_markdown(DOC, "note")
    assert [(c.chunk_id, c.heading, c.text) for c in chunks] == [
        ("note#000", "Title", "Intro words here."),
        ("note#001", "What", "alpha beta gamma"),
        ("note#002", "How", "one two three four five six seven eight nine ten"),
    ]
    assert {c.source for c in chunks} == {"note"}


# Boundary values for section length against max_words=4, overlap=1 (step 3):
# shorter than, equal to, one more than the window, and two full steps.
@pytest.mark.parametrize(
    ("n_words", "expected"),
    [
        (3, ["w0 w1 w2"]),
        (4, ["w0 w1 w2 w3"]),
        (5, ["w0 w1 w2 w3", "w3 w4"]),
        (7, ["w0 w1 w2 w3", "w3 w4 w5 w6"]),
        (8, ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7"]),
    ],
)
def test_long_sections_are_windowed_with_overlap(n_words, expected):
    body = " ".join(f"w{i}" for i in range(n_words))
    chunks = chunk_markdown(f"# H\n\n{body}", "d", max_words=4, overlap_words=1)
    assert [c.text for c in chunks] == expected


def test_text_before_the_first_heading_is_kept_with_empty_heading():
    chunks = chunk_markdown("lead in\n\n# H\n\nbody", "d")
    assert [(c.heading, c.text) for c in chunks] == [("", "lead in"), ("H", "body")]


def test_headings_without_body_produce_no_chunk():
    assert [c.heading for c in chunk_markdown("# Empty\n\n## Full\n\ntext", "d")] == ["Full"]


def test_hash_lines_inside_code_blocks_are_not_headings():
    doc = "# Real\n\n```\n# a comment\nx = 1\n```\n"
    chunks = chunk_markdown(doc, "d")
    assert len(chunks) == 1
    assert "# a comment" in chunks[0].text


@pytest.mark.parametrize("text", ["", "   \n\n", "# Only a heading"])
def test_documents_without_content_give_no_chunks(text):
    assert chunk_markdown(text, "d") == []


@pytest.mark.parametrize(("max_words", "overlap"), [(0, 0), (5, 5), (5, 6), (5, -1)])
def test_invalid_sizes_rejected(max_words, overlap):
    with pytest.raises(ValueError):
        chunk_markdown(DOC, "d", max_words=max_words, overlap_words=overlap)


def test_chunk_directory_reads_files_in_name_order(tmp_path):
    (tmp_path / "b.md").write_text("# B\n\nsecond")
    (tmp_path / "a.md").write_text("# A\n\nfirst")
    (tmp_path / "skip.txt").write_text("# T\n\nignored")
    chunks = chunk_directory(tmp_path)
    assert [(c.chunk_id, c.text) for c in chunks] == [("a#000", "first"), ("b#000", "second")]


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.differing_executors])
@given(
    words=st.lists(st.from_regex(r"[a-z]{1,8}", fullmatch=True), min_size=1, max_size=200),
    max_words=st.integers(2, 40),
    overlap=st.integers(0, 39),
)
def test_no_word_is_lost_and_chunks_respect_the_limit(words, max_words, overlap):
    overlap = min(overlap, max_words - 1)
    chunks = chunk_markdown("# H\n\n" + " ".join(words), "d", max_words, overlap)
    assert all(len(c.text.split()) <= max_words for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    rebuilt = chunks[0].text.split()
    for chunk in chunks[1:]:
        rebuilt.extend(chunk.text.split()[overlap:])
    assert rebuilt == words
    assert chunk_markdown("# H\n\n" + " ".join(words), "d", max_words, overlap) == chunks
