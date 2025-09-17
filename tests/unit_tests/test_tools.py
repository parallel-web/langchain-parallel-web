from __future__ import annotations

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_parallel_web.tools import ParallelWebSearchTool


class TestParallelWebSearchToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[ParallelWebSearchTool]:
        return ParallelWebSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        # Provide test API key to avoid validation error
        return {"api_key": "test-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        """Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {
            "objective": "What are the latest developments in AI?",
            "max_results": 3,
        }
