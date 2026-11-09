from hybrid_rag.chunking import Chunk, clean_line, is_excluded, split_sections, window_words


def test_sections_keep_anchor_and_heading_path() -> None:
    md = "# Title { #title }\nintro\n## Sub Part\nbody\n### Deep { #deep-one }\nmore"
    sections = split_sections("a.md", md)
    assert [s.key for s in sections] == ["a.md#title", "a.md#sub-part", "a.md#deep-one"]
    assert sections[2].heading_path == ("Title", "Sub Part", "Deep")


def test_headings_inside_code_fences_are_text() -> None:
    sections = split_sections("a.md", "# A\n```python\n# comment\n```\n")
    assert len(sections) == 1
    assert "# comment" in sections[0].text


def test_clean_line_drops_includes_and_rewrites_admonitions() -> None:
    assert clean_line("{* ../../docs_src/app.py hl[3] *}") is None
    assert clean_line("/// tip") == "Tip:"
    assert clean_line("///") is None
    assert clean_line('a <dfn title="x">term</dfn> here') == "a term here"


def test_window_words_overlaps_and_covers_text() -> None:
    text = " ".join(str(i) for i in range(100))
    windows = window_words(text, max_words=40, overlap=10)
    assert windows[0].split()[-10:] == windows[1].split()[:10]
    assert windows[-1].split()[-1] == "99"


def test_excluded_paths() -> None:
    assert is_excluded("release-notes.md")
    assert is_excluded("reference/apirouter.md")
    assert not is_excluded("tutorial/cors.md")


def test_chunk_roundtrip() -> None:
    chunk = Chunk("a.md#x:0", "a.md", "x", ("A", "X"), "text")
    assert Chunk.from_dict(chunk.to_dict()) == chunk
    assert chunk.title == "A > X"
