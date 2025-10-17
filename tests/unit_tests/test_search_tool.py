"""Unit tests for Parallel AI Search functionality."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from langchain_parallel_web.search_tool import ParallelWebSearchTool


class TestParallelWebSearchTool:
    """Test cases for ParallelWebSearchTool."""

    def test_tool_initialization(self) -> None:
        """Test tool can be initialized."""
        with patch(
            "langchain_parallel_web.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            assert tool.name == "parallel_web_search"
            assert "Search the web using Parallel AI" in tool.description

    def test_tool_validation_requires_objective_or_queries(self) -> None:
        """Test that tool validates input requirements."""
        with patch(
            "langchain_parallel_web.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()

            with pytest.raises(
                ValueError,
                match="Either 'objective' or 'search_queries' must be provided",
            ):
                tool._run()

    def test_tool_validates_search_queries_limit(self) -> None:
        """Test that tool validates search queries limit."""
        with patch(
            "langchain_parallel_web.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()

            with pytest.raises(ValueError, match="Maximum 5 search queries allowed"):
                tool._run(search_queries=["q1", "q2", "q3", "q4", "q5", "q6"])

    def test_tool_validates_query_length(self) -> None:
        """Test that tool validates individual query length."""
        with patch(
            "langchain_parallel_web.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()

            long_query = "a" * 201
            with pytest.raises(
                ValueError, match="Each search query must be 200 characters or less"
            ):
                tool._run(search_queries=[long_query])

    def test_tool_validates_objective_length(self) -> None:
        """Test that tool validates objective length."""
        with patch(
            "langchain_parallel_web.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()

            long_objective = "a" * 5001
            with pytest.raises(
                ValueError, match="Objective must be 5000 characters or less"
            ):
                tool._run(objective=long_objective)

    def test_tool_validates_max_results_range(self) -> None:
        """Test that tool validates max_results range."""
        with patch(
            "langchain_parallel_web.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()

            with pytest.raises(
                ValueError, match="max_results must be between 1 and 40"
            ):
                tool._run(objective="test", max_results=0)

            with pytest.raises(
                ValueError, match="max_results must be between 1 and 40"
            ):
                tool._run(objective="test", max_results=41)

    def test_tool_validates_max_chars_per_result(self) -> None:
        """Test that tool validates max_chars_per_result minimum."""
        with patch(
            "langchain_parallel_web.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()

            with pytest.raises(
                ValueError, match="max_chars_per_result must be at least 100"
            ):
                tool._run(objective="test", max_chars_per_result=99)

    @patch("langchain_parallel_web.search_tool.get_search_client")
    def test_tool_successful_search(self, mock_get_client: Mock) -> None:
        """Test successful search execution."""
        # Mock the search client
        mock_client = Mock()
        mock_client.search.return_value = {
            "search_id": "test-123",
            "results": [
                {
                    "url": "https://example.com",
                    "title": "Test Result",
                    "excerpts": ["Test excerpt"],
                }
            ],
        }
        mock_get_client.return_value = mock_client

        with patch(
            "langchain_parallel_web.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            result = tool._run(objective="test search")

            assert result["search_id"] == "test-123"
            assert len(result["results"]) == 1
            assert result["results"][0]["title"] == "Test Result"

    @patch("langchain_parallel_web.search_tool.get_search_client")
    def test_tool_handles_api_error(self, mock_get_client: Mock) -> None:
        """Test tool handles API errors gracefully."""
        # Mock the search client to raise an exception
        mock_client = Mock()
        mock_client.search.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        with patch(
            "langchain_parallel_web.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()

            with pytest.raises(
                ValueError, match="Error calling Parallel AI Search API: API Error"
            ):
                tool._run(objective="test search")

    @patch("langchain_parallel_web.search_tool.get_search_client")
    def test_metadata_collection(self, mock_get_client: Mock) -> None:
        """Test metadata collection."""
        mock_client = Mock()
        mock_client.search.return_value = {
            "search_id": "test-123",
            "results": [{"url": "https://example.com", "title": "Test"}],
        }
        mock_get_client.return_value = mock_client

        with patch(
            "langchain_parallel_web.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            result = tool._run(
                search_queries=["query1", "query2"],
                processor="pro",
                include_metadata=True,
            )

            assert "search_metadata" in result
            metadata = result["search_metadata"]
            assert "search_duration_seconds" in metadata
            assert "processor_used" in metadata
            assert metadata["processor_used"] == "pro"
            assert "query_count" in metadata
            assert metadata["query_count"] == 2

    @patch("langchain_parallel_web.search_tool.get_async_search_client")
    async def test_async_functionality(self, mock_get_async_client: Mock) -> None:
        """Test async search functionality."""
        mock_client = Mock()
        mock_client.search = AsyncMock(
            return_value={
                "search_id": "async-test-123",
                "results": [{"url": "https://example.com", "title": "Async Test"}],
            }
        )
        mock_get_async_client.return_value = mock_client

        with patch(
            "langchain_parallel_web.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            result = await tool._arun(objective="test async search")

            assert result["search_id"] == "async-test-123"
            assert len(result["results"]) == 1
