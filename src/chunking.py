"""Splits long document text into smaller overlapping pieces (chunks).

Why: search works best on short, focused passages, and the answer model
has a limit on how much text it can read at once. The overlap means a
sentence that falls on a boundary still appears whole in at least one
chunk, so nothing is lost at the edges.
"""


def chunk_text(
    text: str, chunk_size_words: int = 250, overlap_words: int = 50
) -> list[str]:
    """Splits text into chunks of about chunk_size_words words.

    Each chunk repeats the last overlap_words words of the previous one.
    Very short texts come back as a single chunk. Empty text gives an
    empty list.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be a positive number")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks = []
    step = chunk_size_words - overlap_words
    start = 0
    while start < len(words):
        piece = words[start : start + chunk_size_words]
        chunks.append(" ".join(piece))
        if start + chunk_size_words >= len(words):
            break
        start += step
    return chunks
