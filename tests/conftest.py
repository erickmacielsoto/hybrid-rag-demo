from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from hybrid_rag.chunking import chunk_sections, split_sections
from hybrid_rag.retrieval import HybridRetriever, tokenize

DOCS = {
    "tutorial/cors.md": """# CORS (Cross-Origin Resource Sharing) { #cors }
Intro to cross origin requests from a browser frontend.
## Origin { #origin }
An origin is the combination of protocol, domain and port.
## Use CORSMiddleware { #use-corsmiddleware }
Import CORSMiddleware, create a list of allowed origins and add it as a middleware.
""",
    "tutorial/background-tasks.md": """# Background Tasks { #background-tasks }
Run work after returning a response, for example sending an email notification.
## Using BackgroundTasks { #using-backgroundtasks }
Declare a parameter of type BackgroundTasks and call add_task.
""",
    "advanced/events.md": """# Lifespan Events { #lifespan-events }
Run code once at startup and shutdown, such as loading a machine learning model.
```python
# not a heading
```
""",
}


class HashingEncoder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], *, query: bool) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class OverlapReranker:
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(p)))) for p in passages]


@pytest.fixture
def chunks() -> list:
    sections = [s for file, md in DOCS.items() for s in split_sections(file, md)]
    return chunk_sections(sections)


@pytest.fixture
def retriever(chunks: list) -> HybridRetriever:
    return HybridRetriever(chunks, encoder=HashingEncoder(), reranker=OverlapReranker())
