from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDER, device: str | None = None, batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str], *, query: bool) -> NDArray[np.float32]:
        inputs = [self.query_prefix + t for t in texts] if query else list(texts)
        vectors = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages], show_progress_bar=False)]
