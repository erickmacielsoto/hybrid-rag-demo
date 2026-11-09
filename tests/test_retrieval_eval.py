import pytest

from hybrid_rag.answer import answer, parse_refs
from hybrid_rag.evaluation import GoldItem, evaluate, first_relevant, hit_at_k, mrr
from hybrid_rag.retrieval import BM25, HybridRetriever, Mode, rrf


def test_bm25_prefers_documents_with_query_terms() -> None:
    scores = BM25(["cats purr loudly", "dogs bark", "cats and dogs"]).scores("cats purr")
    assert scores.argmax() == 0
    assert scores[1] == 0


def test_rrf_rewards_agreement() -> None:
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert fused[1] == fused[2] > fused[3] > 0
    assert 4 in fused


@pytest.mark.parametrize("mode", list(Mode))
def test_every_mode_finds_cors_origin(retriever: HybridRetriever, mode: Mode) -> None:
    hits = retriever.search("what is an origin protocol domain port", k=3, mode=mode)
    assert hits[0].chunk.section_key == "tutorial/cors.md#origin"


def test_results_are_distinct_sections(retriever: HybridRetriever) -> None:
    hits = retriever.search("middleware origins", k=10, mode=Mode.HYBRID)
    keys = [h.chunk.section_key for h in hits]
    assert len(keys) == len(set(keys))


def test_dense_without_encoder_fails(chunks: list) -> None:
    with pytest.raises(ValueError):
        HybridRetriever(chunks).search("x", mode=Mode.DENSE)


def test_metrics() -> None:
    assert first_relevant(["a", "b", "c"], frozenset({"c"})) == 3
    assert first_relevant(["a"], frozenset({"z"})) is None
    ranks = [1, 3, None, 2]
    assert hit_at_k(ranks, 1) == 0.25
    assert hit_at_k(ranks, 3) == 0.75
    assert mrr(ranks) == pytest.approx((1 + 1 / 3 + 1 / 2) / 4)


def test_evaluate_reports_every_k(retriever: HybridRetriever) -> None:
    gold = [
        GoldItem("q1", "send an email after the response", frozenset({"tutorial/background-tasks.md#background-tasks"})),
        GoldItem("q2", "load a machine learning model at startup", frozenset({"advanced/events.md#lifespan-events"})),
    ]
    result = evaluate(retriever, gold, Mode.BM25)
    assert result.hit_at[1] == 1.0
    assert result.mrr_at_10 == 1.0
    assert result.file_hit_at_5 == 1.0


def test_parse_refs_flags_invalid_citations() -> None:
    assert parse_refs("Use X [1], see also [2, 5].", available=3) == ([1, 2], [5])


class EchoGenerator:
    def generate(self, system: str, user: str) -> str:
        assert "[1] tutorial/cors.md" in user
        return "Add CORSMiddleware with your allowed origins [1][7]."


def test_answer_with_generator_keeps_citations(retriever: HybridRetriever) -> None:
    hits = retriever.search("allow origins middleware", k=2)
    result = answer("How do I enable CORS?", hits, EchoGenerator())
    assert result.cited_refs == [1]
    assert result.invalid_refs == [7]
    assert result.citations[0].render().startswith("[1] tutorial/cors.md § ")


def test_extractive_answer_and_empty_case(retriever: HybridRetriever) -> None:
    hits = retriever.search("background email", k=2)
    assert answer("q", hits).text.endswith("[1]")
    assert answer("q", []).citations == []
