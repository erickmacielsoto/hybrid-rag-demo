import pytest
from mcp import Client

from hybrid_rag.mcp_server import create_server
from hybrid_rag.retrieval import HybridRetriever

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_lists_three_tools(retriever: HybridRetriever) -> None:
    async with Client(create_server(retriever)) as client:
        tools = await client.list_tools()
        assert sorted(t.name for t in tools.tools) == ["answer_question", "get_section", "search_docs"]


async def test_search_returns_citations(retriever: HybridRetriever) -> None:
    async with Client(create_server(retriever)) as client:
        result = await client.call_tool("search_docs", {"query": "origin protocol domain port", "k": 2})
        assert not result.is_error
        hits = result.structured_content["hits"]
        assert hits[0]["citation"] == "[1] tutorial/cors.md § CORS (Cross-Origin Resource Sharing) > Origin"


async def test_get_section_and_unknown_section(retriever: HybridRetriever) -> None:
    async with Client(create_server(retriever)) as client:
        ok = await client.call_tool("get_section", {"file": "tutorial/cors.md", "anchor": "origin"})
        assert "protocol, domain and port" in ok.structured_content["text"]
        missing = await client.call_tool("get_section", {"file": "tutorial/cors.md", "anchor": "nope"})
        assert missing.is_error


async def test_validation_rejects_bad_k(retriever: HybridRetriever) -> None:
    async with Client(create_server(retriever)) as client:
        result = await client.call_tool("search_docs", {"query": "cors", "k": 99})
        assert result.is_error


async def test_answer_question_builds_grounded_prompt(retriever: HybridRetriever) -> None:
    async with Client(create_server(retriever)) as client:
        result = await client.call_tool("answer_question", {"question": "How do I run work after the response?"})
        payload = result.structured_content
        assert payload["sources"][0].startswith("[1] tutorial/background-tasks.md")
        assert "Question: How do I run work after the response?" in payload["prompt"]
