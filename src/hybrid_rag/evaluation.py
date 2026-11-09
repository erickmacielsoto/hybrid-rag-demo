from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .retrieval import HybridRetriever, Mode

KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class GoldItem:
    id: str
    question: str
    relevant: frozenset[str]


@dataclass(frozen=True)
class ModeResult:
    mode: Mode
    label: str
    n: int
    hit_at: dict[int, float]
    mrr_at_10: float
    file_hit_at_5: float
    latency_ms_mean: float
    latency_ms_p95: float
    first_relevant_rank: dict[str, int | None]


def load_gold(path: Path) -> list[GoldItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            items.append(GoldItem(row["id"], row["question"], frozenset(row["relevant"])))
    return items


def first_relevant(ranked_keys: Sequence[str], relevant: frozenset[str]) -> int | None:
    for rank, key in enumerate(ranked_keys, start=1):
        if key in relevant:
            return rank
    return None


def hit_at_k(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: Sequence[int | None], cutoff: int = 10) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= cutoff) / len(ranks) if ranks else 0.0


def evaluate(
    retriever: HybridRetriever, gold: Sequence[GoldItem], mode: Mode, depth: int = 10, label: str | None = None
) -> ModeResult:
    ranks: dict[str, int | None] = {}
    file_ranks: list[int | None] = []
    latencies: list[float] = []
    for item in gold:
        start = time.perf_counter()
        hits = retriever.search(item.question, k=depth, mode=mode)
        latencies.append((time.perf_counter() - start) * 1000)
        ranks[item.id] = first_relevant([h.chunk.section_key for h in hits], item.relevant)
        relevant_files = frozenset(key.split("#")[0] for key in item.relevant)
        file_ranks.append(first_relevant([h.chunk.file for h in hits], relevant_files))
    values = list(ranks.values())
    ordered = sorted(latencies)
    return ModeResult(
        mode=mode,
        label=label or mode.value,
        n=len(gold),
        hit_at={k: hit_at_k(values, k) for k in KS},
        mrr_at_10=mrr(values),
        file_hit_at_5=hit_at_k(file_ranks, 5),
        latency_ms_mean=sum(latencies) / len(latencies) if latencies else 0.0,
        latency_ms_p95=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0.0,
        first_relevant_rank=ranks,
    )


def to_markdown(results: Sequence[ModeResult]) -> str:
    header = "| Retriever | n | " + " | ".join(f"Hit@{k}" for k in KS) + " | MRR@10 | File Hit@5 | Mean latency ms | p95 ms |"
    sep = "|---|---:|" + "---:|" * len(KS) + "---:|---:|---:|---:|"
    rows = [
        f"| {r.label} | {r.n} | "
        + " | ".join(f"{r.hit_at[k]:.3f}" for k in KS)
        + f" | {r.mrr_at_10:.3f} | {r.file_hit_at_5:.3f} | {r.latency_ms_mean:.1f} | {r.latency_ms_p95:.1f} |"
        for r in results
    ]
    return "\n".join([header, sep, *rows]) + "\n"


def to_json(results: Sequence[ModeResult]) -> list[dict[str, object]]:
    return [
        {
            "mode": r.mode.value,
            "label": r.label,
            "n": r.n,
            "hit_at": {str(k): v for k, v in r.hit_at.items()},
            "mrr_at_10": r.mrr_at_10,
            "file_hit_at_5": r.file_hit_at_5,
            "latency_ms_mean": r.latency_ms_mean,
            "latency_ms_p95": r.latency_ms_p95,
            "first_relevant_rank": r.first_relevant_rank,
        }
        for r in results
    ]
