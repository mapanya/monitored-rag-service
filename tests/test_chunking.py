"""Tests for the text splitting logic."""

import pytest

from src.chunking import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_stays_whole():
    text = "A short document that fits in one chunk."
    assert chunk_text(text) == [text]


def test_long_text_is_split_with_overlap():
    words = [f"word{i}" for i in range(600)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size_words=250, overlap_words=50)

    assert len(chunks) == 3
    # Overlap: the second chunk starts 200 words in (250 - 50).
    assert chunks[1].split()[0] == "word200"
    # No chunk is longer than the chosen size.
    assert all(len(c.split()) <= 250 for c in chunks)
    # Nothing is lost: the last word survives into the final chunk.
    assert chunks[-1].split()[-1] == "word599"


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size_words=0)
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size_words=100, overlap_words=100)
