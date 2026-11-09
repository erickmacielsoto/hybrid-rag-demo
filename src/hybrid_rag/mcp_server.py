from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .answer import build_prompt, citations_for
from .retrieval import HybridRetriever, Mode


class SearchHit(BaseModel):
    citation: str
    file: str
    section: str
    anchor: str
    score: float
    text: str


class SearchResult(BaseModel):
    mode: str
    hits: list[SearchHit]


class SectionText(BaseModel):
    file: str
    anchor: str
    section: str
    text: str


class GroundedContext(BaseModel):
    question: str
    sources: list[str]
    prompt: str
    instructions: str


READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def create_server(retriever: HybridRetriever) -> MCPServer:
    server = MCPServer(
        name="fastapi-docs-rag",
        version="0.1.0",
        instructions="Hybrid (BM25 + dense + cross-encoder) retrieval over the FastAPI documentation. "
        "Cite results as file § section.",
    )
    sections: dict[str, list[str]] = {}
    titles: dict[str, str] = {}
    for chunk in retriever.chunks:
        sections.setdefault(chunk.section_key, []).append(chunk.text)
        titles[chunk.section_key] = chunk.title

    @server.tool(annotations=READ_ONLY)
    def search_docs(
        query: Annotated[str, Field(min_length=2, max_length=500)],
        k: Annotated[int, Field(ge=1, le=20)] = 5,
        mode: Mode = Mode.HYBRID_RERANK,
    ) -> SearchResult:
        """Search the FastAPI docs. Returns the k best distinct sections with file and section citations."""
        hits = retriever.search(query, k=k, mode=mode)
        return SearchResult(
            mode=mode.value,
            hits=[
                SearchHit(
                    citation=c.render(),
                    file=h.chunk.file,
                    section=h.chunk.title,
                    anchor=h.chunk.anchor,
                    score=round(h.score, 4),
                    text=h.chunk.text,
                )
                for h, c in zip(hits, citations_for(hits), strict=True)
            ],
        )

    @server.tool(annotations=READ_ONLY)
    def get_section(
        file: Annotated[str, Field(min_length=4, pattern=r"\.md$")],
        anchor: Annotated[str, Field(min_length=1, pattern=r"^[\w-]+$")],
    ) -> SectionText:
        """Return the full text of one documentation section, e.g. file='tutorial/cors.md', anchor='use-corsmiddleware'."""
        key = f"{file}#{anchor}"
        if key not in sections:
            raise ToolError(f"Unknown section {key}")
        return SectionText(file=file, anchor=anchor, section=titles[key], text="\n\n".join(sections[key]))

    @server.tool(annotations=READ_ONLY)
    def answer_question(
        question: Annotated[str, Field(min_length=3, max_length=1000)],
        k: Annotated[int, Field(ge=1, le=8)] = 4,
    ) -> GroundedContext:
        """Retrieve sources for a question and return a numbered, citation-ready context for the calling model."""
        hits = retriever.search(question, k=k, mode=Mode.HYBRID_RERANK)
        return GroundedContext(
            question=question,
            sources=[c.render() for c in citations_for(hits)],
            prompt=build_prompt(question, hits),
            instructions="Answer only from the sources and cite them as [n]; say so if they are insufficient.",
        )

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the hybrid RAG index over MCP (stdio).")
    parser.add_argument("--index", type=Path, default=Path("index"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    from .store import load_retriever

    create_server(load_retriever(args.index, device=args.device)).run("stdio")


if __name__ == "__main__":
    main()
