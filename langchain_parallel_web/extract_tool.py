"""Parallel AI Extract Tool for LangChain."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr, model_validator

from ._client import get_api_key, get_extract_client


class ParallelExtractInput(BaseModel):
    """Input schema for Parallel AI Extract Tool."""

    urls: list[str] = Field(description="List of URLs to extract content from")


class ParallelExtractTool(BaseTool):
    """Parallel AI Extract Tool.

    This tool extracts clean, structured content from web pages using the
    Parallel AI Extract API.

    Setup:
        Install ``langchain-parallel-web`` and set environment variable
        ``PARALLEL_AI_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-parallel-web
            export PARALLEL_AI_API_KEY="your-api-key"

    Key init args:
        api_key: Optional[SecretStr]
            Parallel AI API key. If not provided, will be read from
            PARALLEL_AI_API_KEY env var.
        base_url: str
            Base URL for Parallel AI API. Defaults to "https://api.parallel.ai".
        max_chars_per_extract: Optional[int]
            Maximum characters per extracted result.

    Instantiation:
        .. code-block:: python

            from langchain_parallel_web import ParallelExtractTool

            # Basic instantiation
            tool = ParallelExtractTool()

            # With custom API key and parameters
            tool = ParallelExtractTool(
                api_key="your-api-key",
                max_chars_per_extract=5000
            )

    Invocation:
        .. code-block:: python

            # Extract content from URLs
            result = tool.invoke({
                "urls": [
                    "https://example.com/article1",
                    "https://example.com/article2"
                ]
            })

            # Result is a list of dicts with url, title, and content
            for item in result:
                print(f"Title: {item['title']}")
                print(f"URL: {item['url']}")
                print(f"Content: {item['content'][:200]}...")

    Response Format:
        Returns a list of dictionaries, each containing:
        - url: The URL that was extracted
        - title: Title of the webpage
        - content: Full extracted content as markdown
        - publish_date: Publish date if available (optional)
    """

    name: str = "parallel_extract"
    description: str = (
        "Extract clean, structured content from web pages. "
        "Input should be a list of URLs to extract content from. "
        "Returns extracted content formatted as markdown."
    )
    args_schema: type[BaseModel] = ParallelExtractInput

    api_key: Optional[SecretStr] = Field(default=None)
    """Parallel AI API key. If not provided, will be read from env var."""

    base_url: str = Field(default="https://api.parallel.ai")
    """Base URL for Parallel AI API."""

    max_chars_per_extract: Optional[int] = None
    """Maximum characters per extracted result."""

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        # Get API key from parameter or environment
        api_key = values.get("api_key")
        if isinstance(api_key, SecretStr):
            api_key_str: Optional[str] = api_key.get_secret_value()
        else:
            api_key_str = api_key

        # This will raise an error if API key is not found
        get_api_key(api_key_str)

        return values

    def _run(
        self,
        urls: list[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list[dict[str, Any]]:
        """Extract content from URLs.

        Args:
            urls: List of URLs to extract content from
            run_manager: Callback manager for the tool run

        Returns:
            List of dictionaries with extracted content
        """
        try:
            # Get API key
            api_key_str = get_api_key(
                self.api_key.get_secret_value() if self.api_key else None
            )

            # Initialize extract client
            client = get_extract_client(api_key_str, self.base_url)

            # Build full_content config if max_chars is specified
            full_content = None
            if self.max_chars_per_extract:
                full_content = {"max_characters": self.max_chars_per_extract}

            # Extract content from URLs
            extract_response = client.extract(
                urls=urls,
                full_content=full_content,
            )

            results = extract_response.get("results", [])
            errors = extract_response.get("errors", [])

            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "url": result.get("url"),
                    "title": result.get("title"),
                    "content": result.get("full_content", ""),
                }

                # Add optional fields if present
                if "publish_date" in result:
                    formatted_result["publish_date"] = result["publish_date"]

                formatted_results.append(formatted_result)

            # If there were errors, add them to the results with error info
            formatted_results.extend(
                [
                    {
                        "url": error.get("url"),
                        "title": None,
                        "content": f"Error: {error.get('message', 'Unknown error')}",
                        "error_type": error.get("error_type"),
                    }
                    for error in errors
                ]
            )

            return formatted_results

        except Exception as e:
            msg = f"Error calling Parallel AI Extract API: {e!s}"
            raise ValueError(msg) from e

    async def _arun(
        self,
        urls: list[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list[dict[str, Any]]:
        """Extract content from URLs asynchronously.

        Args:
            urls: List of URLs to extract content from
            run_manager: Callback manager for the tool run

        Returns:
            List of dictionaries with extracted content
        """
        from ._client import get_async_extract_client

        try:
            # Get API key
            api_key_str = get_api_key(
                self.api_key.get_secret_value() if self.api_key else None
            )

            # Initialize async extract client
            client = get_async_extract_client(api_key_str, self.base_url)

            # Build full_content config if max_chars is specified
            full_content = None
            if self.max_chars_per_extract:
                full_content = {"max_characters": self.max_chars_per_extract}

            # Extract content from URLs
            extract_response = await client.extract(
                urls=urls,
                full_content=full_content,
            )

            results = extract_response.get("results", [])
            errors = extract_response.get("errors", [])

            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "url": result.get("url"),
                    "title": result.get("title"),
                    "content": result.get("full_content", ""),
                }

                # Add optional fields if present
                if "publish_date" in result:
                    formatted_result["publish_date"] = result["publish_date"]

                formatted_results.append(formatted_result)

            # If there were errors, add them to the results with error info
            formatted_results.extend(
                [
                    {
                        "url": error.get("url"),
                        "title": None,
                        "content": f"Error: {error.get('message', 'Unknown error')}",
                        "error_type": error.get("error_type"),
                    }
                    for error in errors
                ]
            )

            return formatted_results

        except Exception as e:
            msg = f"Error calling Parallel AI Extract API: {e!s}"
            raise ValueError(msg) from e
