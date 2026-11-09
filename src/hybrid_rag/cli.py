from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer import HFGenerator, answer
from .evaluation import evaluate, load_gold, to_json, to_markdown
from .models import CrossEncoderReranker
from .retrieval import Mode
from .store import build_index, load_retriever


def _build(args: argparse.Namespace) -> None:
    meta = build_index(args.docs, args.index, device=args.device)
    print(json.dumps(meta.__dict__, indent=2))


def _search(args: argparse.Namespace) -> None:
    retriever = load_retriever(args.index, device=args.device)
    for i, hit in enumerate(retriever.search(args.query, k=args.k, mode=Mode(args.mode)), start=1):
        print(f"[{i}] {hit.score:.3f} {hit.chunk.file} § {hit.chunk.title}")


def _ask(args: argparse.Namespace) -> None:
    retriever = load_retriever(args.index, device=args.device)
    generator = HFGenerator(args.generator) if args.generator else None
    for question in args.questions:
        hits = retriever.search(question, k=args.k)
        result = answer(question, hits, generator)
        print(f"## {question}\n\n{result.render()}\n")


def _eval(args: argparse.Namespace) -> None:
    gold = load_gold(args.gold)
    rerankers = args.rerankers or [None]
    retriever = load_retriever(args.index, device=args.device, reranker=rerankers[0])
    retriever.search("warm up", k=1, mode=Mode.HYBRID_RERANK)
    results = []
    for mode in (Mode.BM25, Mode.DENSE, Mode.HYBRID):
        results.append(evaluate(retriever, gold, mode))
    for name in rerankers:
        if name is not None:
            retriever.reranker = CrossEncoderReranker(name, device=args.device)
            retriever.search("warm up", k=1, mode=Mode.HYBRID_RERANK)
        label = f"hybrid_rerank ({name.split('/')[-1]})" if name else Mode.HYBRID_RERANK.value
        results.append(evaluate(retriever, gold, Mode.HYBRID_RERANK, label=label))
    for r in results:
        print(f"{r.label}: hit@5={r.hit_at[5]:.3f} mrr={r.mrr_at_10:.3f}", flush=True)
    markdown = to_markdown(results)
    print(markdown)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "retrieval_eval.md").write_text(markdown, encoding="utf-8")
    (args.out / "retrieval_eval.json").write_text(json.dumps(to_json(results), indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hybrid-rag")
    parser.add_argument("--index", type=Path, default=Path("index"))
    parser.add_argument("--device", default=None, help="cpu, cuda or None for auto")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="Chunk the docs and embed them.")
    p.add_argument("--docs", type=Path, required=True)
    p.set_defaults(func=_build)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.HYBRID_RERANK.value)
    p.set_defaults(func=_search)

    p = sub.add_parser("ask", help="Answer with citations; --generator enables a local LLM.")
    p.add_argument("questions", nargs="+")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--generator", default=None, help="e.g. Qwen/Qwen2.5-1.5B-Instruct")
    p.set_defaults(func=_ask)

    p = sub.add_parser("eval", help="Hit@k and MRR for every retrieval mode.")
    p.add_argument("--gold", type=Path, default=Path("data/gold.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--rerankers", nargs="*", default=None, help="cross-encoder ids to compare")
    p.set_defaults(func=_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
