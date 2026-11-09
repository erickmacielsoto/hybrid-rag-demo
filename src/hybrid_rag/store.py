from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, load_corpus
from .models import DEFAULT_EMBEDDER, DEFAULT_RERANKER, CrossEncoderReranker, SentenceTransformerEncoder
from .retrieval import HybridRetriever


@dataclass(frozen=True)
class IndexMeta:
    docs_root: str
    embedder: str
    reranker: str
    chunks: int
    sections: int
    files: int
    build_seconds: float


def build_index(docs_root: Path, index_dir: Path, embedder: str = DEFAULT_EMBEDDER, device: str | None = None) -> IndexMeta:
    start = time.perf_counter()
    chunks = load_corpus(docs_root)
    encoder = SentenceTransformerEncoder(embedder, device=device)
    embeddings = encoder.encode([c.embed_text() for c in chunks], query=False)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c.to_dict()) for c in chunks) + "\n", encoding="utf-8"
    )
    meta = IndexMeta(
        docs_root=docs_root.as_posix(),
        embedder=embedder,
        reranker=DEFAULT_RERANKER,
        chunks=len(chunks),
        sections=len({c.section_key for c in chunks}),
        files=len({c.file for c in chunks}),
        build_seconds=round(time.perf_counter() - start, 1),
    )
    (index_dir / "meta.json").write_text(json.dumps(meta.__dict__, indent=2), encoding="utf-8")
    return meta


def load_retriever(
    index_dir: Path, device: str | None = None, with_reranker: bool = True, reranker: str | None = None
) -> HybridRetriever:
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    lines = (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    chunks = [Chunk.from_dict(json.loads(line)) for line in lines if line]
    embeddings = np.load(index_dir / "embeddings.npy")
    return HybridRetriever(
        chunks,
        encoder=SentenceTransformerEncoder(meta["embedder"], device=device),
        reranker=CrossEncoderReranker(reranker or meta["reranker"], device=device) if with_reranker else None,
        embeddings=embeddings,
    )
