from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
        "i", "if", "in", "is", "it", "its", "my", "of", "on", "or", "should", "that", "the", "this",
        "to", "what", "when", "where", "which", "why", "with", "you", "your",
    }
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class Mode(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], *, query: bool) -> NDArray[np.float32]: ...


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in documents]
        self.lengths = np.array([sum(tf.values()) for tf in self.tf], dtype=np.float64)
        self.avg = float(self.lengths.mean()) if len(documents) else 0.0
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(documents)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> NDArray[np.float64]:
        out = np.zeros(len(self.tf))
        terms = set(tokenize(query))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
            out[i] = sum(
                self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
        return out


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def dedupe_sections(hits: Sequence[Hit], k: int) -> list[Hit]:
    seen: set[str] = set()
    out: list[Hit] = []
    for hit in hits:
        if hit.chunk.section_key in seen:
            continue
        seen.add(hit.chunk.section_key)
        out.append(hit)
        if len(out) == k:
            break
    return out


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        encoder: Encoder | None = None,
        reranker: Reranker | None = None,
        embeddings: NDArray[np.float32] | None = None,
        candidates: int = 50,
        rerank_top: int = 30,
    ) -> None:
        self.chunks = list(chunks)
        self.bm25 = BM25([c.embed_text() for c in self.chunks])
        self.encoder = encoder
        self.reranker = reranker
        self.candidates = candidates
        self.rerank_top = rerank_top
        if embeddings is None and encoder is not None:
            embeddings = encoder.encode([c.embed_text() for c in self.chunks], query=False)
        self.embeddings = embeddings

    def _bm25_ranking(self, query: str) -> tuple[list[int], NDArray[np.float64]]:
        scores = self.bm25.scores(query)
        order = np.argsort(-scores, kind="stable")[: self.candidates]
        return [int(i) for i in order if scores[i] > 0], scores

    def _dense_ranking(self, query: str) -> tuple[list[int], NDArray[np.float32]]:
        if self.encoder is None or self.embeddings is None:
            raise ValueError("dense retrieval requires an encoder")
        q = self.encoder.encode([query], query=True)[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores, kind="stable")[: self.candidates]
        return [int(i) for i in order], scores

    def search(self, query: str, k: int = 5, mode: Mode = Mode.HYBRID_RERANK) -> list[Hit]:
        if mode is Mode.BM25:
            ranking, bm = self._bm25_ranking(query)
            hits = [Hit(self.chunks[i], float(bm[i])) for i in ranking]
        elif mode is Mode.DENSE:
            ranking, dense = self._dense_ranking(query)
            hits = [Hit(self.chunks[i], float(dense[i])) for i in ranking]
        else:
            fused = rrf([self._bm25_ranking(query)[0], self._dense_ranking(query)[0]])
            ordered = sorted(fused.items(), key=lambda kv: -kv[1])
            hits = [Hit(self.chunks[i], score) for i, score in ordered]
            if mode is Mode.HYBRID_RERANK:
                if self.reranker is None:
                    raise ValueError("hybrid_rerank requires a reranker")
                pool = dedupe_sections(hits, self.rerank_top)
                scores = self.reranker.score(query, [h.chunk.embed_text() for h in pool])
                hits = sorted(
                    (Hit(h.chunk, s) for h, s in zip(pool, scores, strict=True)), key=lambda h: -h.score
                )
        return dedupe_sections(hits, k)
