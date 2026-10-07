"""Split Markdown documents into overlapping, heading-aware text chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    """A piece of a document small enough to embed and retrieve."""

    chunk_id: str
    source: str
    heading: str
    text: str


def chunk_markdown(
    text: str, source: str, max_words: int = 160, overlap_words: int = 30
) -> list[Chunk]:
    """Split one Markdown document into chunks.

    Sections start at headings. A section longer than ``max_words`` is cut
    into windows that share ``overlap_words`` with the previous window, so a
    sentence on a boundary is not lost. Chunk IDs are ``<source>#<nnn>``.
    """
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be >= 0 and smaller than max_words")

    chunks: list[Chunk] = []
    for heading, body in _sections(text):
        words = body.split()
        step = max_words - overlap_words
        for start in range(0, len(words), step):
            window = words[start : start + max_words]
            chunks.append(Chunk(f"{source}#{len(chunks):03d}", source, heading, " ".join(window)))
            if start + max_words >= len(words):
                break
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs; text before the first heading has an empty heading."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            sections.append((match.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(heading, "\n".join(lines)) for heading, lines in sections if "".join(lines).strip()]


def chunk_directory(directory: str | Path, pattern: str = "*.md", **kwargs) -> list[Chunk]:
    """Chunk every matching file in ``directory``, in file-name order."""
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob(pattern)):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.stem, **kwargs))
    return chunks
