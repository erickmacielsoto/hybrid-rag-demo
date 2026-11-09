from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXCLUDED = (
    "release-notes.md",
    "reference/",
    "fastapi-people.md",
    "management.md",
    "management-tasks.md",
    "external-links.md",
    "newsletter.md",
    "translation-banner.md",
    "agent-banner.md",
    "_llm-test.md",
    "contributing.md",
)

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{\s*#([\w-]+)\s*\})?\s*$")
_INCLUDE = re.compile(r"^\{\*.*\*\}\s*$")
_ADMONITION = re.compile(r"^///\s*(\w+)?.*$")
_HTML_TAG = re.compile(r"</?(?:dfn|abbr|span|div|a|img|b|i|em|strong)\b[^>]*>")


@dataclass(frozen=True)
class Section:
    file: str
    anchor: str
    heading_path: tuple[str, ...]
    text: str

    @property
    def key(self) -> str:
        return f"{self.file}#{self.anchor}"


@dataclass(frozen=True)
class Chunk:
    id: str
    file: str
    anchor: str
    heading_path: tuple[str, ...]
    text: str

    @property
    def section_key(self) -> str:
        return f"{self.file}#{self.anchor}"

    @property
    def title(self) -> str:
        return " > ".join(self.heading_path)

    def embed_text(self) -> str:
        return f"{self.title}\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["heading_path"] = list(self.heading_path)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(**{**data, "heading_path": tuple(data["heading_path"])})


def slugify(title: str) -> str:
    return re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", title.lower()).strip())


def clean_line(line: str) -> str | None:
    if _INCLUDE.match(line.strip()):
        return None
    admonition = _ADMONITION.match(line.strip())
    if admonition:
        kind = admonition.group(1)
        return f"{kind.capitalize()}:" if kind else None
    return _HTML_TAG.sub("", line)


def split_sections(file: str, markdown: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[str] = []
    anchor: str = "top"
    path: tuple[str, ...] = (file,)
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(file, anchor, path, text))

    for raw in markdown.splitlines():
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(raw)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2).strip()
            stack = stack[: level - 1] + [""] * max(0, level - 1 - len(stack)) + [title]
            anchor = heading.group(3) or slugify(title)
            path = tuple(part for part in stack if part)
            buf = []
            continue
        line = raw if in_fence else clean_line(raw)
        if line is not None:
            buf.append(line)
    flush()
    return sections


def window_words(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max(1, max_words - overlap)
    windows = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return windows


def chunk_sections(sections: Iterable[Section], max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        for i, piece in enumerate(window_words(section.text, max_words, overlap)):
            chunks.append(Chunk(f"{section.key}:{i}", section.file, section.anchor, section.heading_path, piece))
    return chunks


def is_excluded(rel_path: str) -> bool:
    return any(rel_path == pattern or rel_path.startswith(pattern) for pattern in EXCLUDED)


def load_corpus(root: Path, max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if is_excluded(rel):
            continue
        sections = split_sections(rel, path.read_text(encoding="utf-8"))
        chunks.extend(chunk_sections(sections, max_words, overlap))
    return chunks
