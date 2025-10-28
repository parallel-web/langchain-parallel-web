"""Parallel Extract Tool for LangChain."""

from __future__ import annotations

from typing import Any, Optional, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr, model_validator

from ._client import get_api_key, get_extract_client


class ExcerptSettings(BaseModel):
    """Settings for excerpt extraction."""

    max_chars_per_result: Optional[int] = Field(
        default=None,
        description=(
            "Optional upper bound on the total number of characters to include "
            "across all excerpts for each url."
        ),
    )


class FullContentSettings(BaseModel):
    """Settings for full content extraction."""

    max_chars_per_result: Optional[int] = Field(
        default=None,
        description=(
            "Optional limit on the number of characters to include in the full "
            "content for each url."
        ),
    )


class FetchPolicy(BaseModel):
    """Fetch policy for cache vs live content."""

    max_age_seconds: Optional[int] = Field(
        default=None,
        description=(
            "Maximum age of cached content in seconds. Minimum 600 seconds. "
            "If not provided, dynamic age policy will be used."
        ),
    )
    timeout_seconds: Optional[float] = Field(
        default=None,
        description=(
            "Timeout in seconds for fetching live content. If unspecified, "
            "dynamic timeout will be used (15-60 seconds)."
        ),
    )
    disable_cache_fallback: bool = Field(
        default=False,
        description=(
            "If false, fallback to cached content if live fetch fails. "
            "If true, returns an error instead."
        ),
    )


class ParallelExtractInput(BaseModel):
    """Input schema for Parallel Extract Tool."""

    urls: list[str] = Field(description="List of URLs to extract content from")

    search_objective: Optional[str] = Field(
        default=None,
        description=(
            "If provided, focuses extracted content on the specified search objective"
        ),
    )

    search_queries: Optional[list[str]] = Field(
        default=None,
        description=(
            "If provided, focuses extracted content on the specified keyword search "
            "queries"
        ),
    )

    excerpts: Union[bool, ExcerptSettings] = Field(
        default=True,
        description=(
            "Include excerpts from each URL relevant to the search objective and "
            "queries. Can be boolean or ExcerptSettings object."
        ),
    )

    full_content: Union[bool, FullContentSettings] = Field(
        default=False,
        description=(
            "Include full content from each URL. Can be boolean or "
            "FullContentSettings object."
        ),
    )

    fetch_policy: Optional[FetchPolicy] = Field(
        default=None,
        description=(
            "Fetch policy: determines when to return content from the cache "
            "(faster) vs fetching live content (fresher)"
        ),
    )


class ParallelExtractTool(BaseTool):
    """Parallel Extract Tool.

    This tool extracts clean, structured content from web pages using the
    Parallel Extract API.

    Setup:
        Install ``langchain-parallel-web`` and set environment variable
        ``PARALLEL_AI_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-parallel-web
            export PARALLEL_AI_API_KEY="your-api-key"

    Key init args:
        api_key: Optional[SecretStr]
            Parallel API key. If not provided, will be read from
            PARALLEL_AI_API_KEY env var.
        base_url: str
            Base URL for Parallel API. Defaults to "https://api.parallel.ai".
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
    """Parallel API key. If not provided, will be read from env var."""

    base_url: str = Field(default="https://api.parallel.ai")
    """Base URL for Parallel API."""

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
        search_objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        excerpts: Union[bool, ExcerptSettings] = True,
        full_content: Union[bool, FullContentSettings] = False,
        fetch_policy: Optional[FetchPolicy] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list[dict[str, Any]]:
        """Extract content from URLs.

        Args:
            urls: List of URLs to extract content from
            search_objective: Optional search objective to focus extraction
            search_queries: Optional keyword search queries to focus extraction
            excerpts: Include excerpts (boolean or ExcerptSettings)
            full_content: Include full content (boolean or FullContentSettings)
            fetch_policy: Optional fetch policy for cache vs live content
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

            # Build full_content config
            full_content_param = full_content
            if self.max_chars_per_extract and isinstance(full_content, bool):
                # Use tool-level config if full_content is just a boolean
                full_content_param = {
                    "max_chars_per_result": self.max_chars_per_extract
                }
            elif isinstance(full_content, FullContentSettings):
                full_content_param = full_content.model_dump(exclude_none=True)

            # Build excerpts config
            excerpts_param = excerpts
            if isinstance(excerpts, ExcerptSettings):
                excerpts_param = excerpts.model_dump(exclude_none=True)

            # Build fetch_policy config
            fetch_policy_param = None
            if fetch_policy:
                fetch_policy_param = fetch_policy.model_dump(exclude_none=True)

            # Extract content from URLs
            extract_response = client.extract(
                urls=urls,
                objective=search_objective,
                search_queries=search_queries,
                excerpts=excerpts_param,
                full_content=full_content_param,
                fetch_policy=fetch_policy_param,
            )

            results = extract_response.get("results", [])
            errors = extract_response.get("errors", [])

            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "url": result.get("url"),
                    "title": result.get("title"),
                }

                # Add excerpts if present
                if "excerpts" in result and result["excerpts"] is not None:
                    formatted_result["excerpts"] = result["excerpts"]
                    # Combine excerpts into content field for backward compatibility
                    if "full_content" not in result:
                        # Excerpts are a list of strings, join them with newlines
                        formatted_result["content"] = "\n\n".join(result["excerpts"])

                # Add full_content if present
                if "full_content" in result:
                    formatted_result["full_content"] = result["full_content"]
                    # For backward compatibility, also set as "content"
                    formatted_result["content"] = result["full_content"]

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
                        "content": f"Error: {error.get('error_type', 'Unknown error')}",
                        "error_type": error.get("error_type"),
                        "http_status_code": error.get("http_status_code"),
                    }
                    for error in errors
                ]
            )

            return formatted_results

        except Exception as e:
            msg = f"Error calling Parallel Extract API: {e!s}"
            raise ValueError(msg) from e

    async def _arun(
        self,
        urls: list[str],
        search_objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        excerpts: Union[bool, ExcerptSettings] = True,
        full_content: Union[bool, FullContentSettings] = False,
        fetch_policy: Optional[FetchPolicy] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list[dict[str, Any]]:
        """Extract content from URLs asynchronously.

        Args:
            urls: List of URLs to extract content from
            search_objective: Optional search objective to focus extraction
            search_queries: Optional keyword search queries to focus extraction
            excerpts: Include excerpts (boolean or ExcerptSettings)
            full_content: Include full content (boolean or FullContentSettings)
            fetch_policy: Optional fetch policy for cache vs live content
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

            # Build full_content config
            full_content_param = full_content
            if self.max_chars_per_extract and isinstance(full_content, bool):
                # Use tool-level config if full_content is just a boolean
                full_content_param = {
                    "max_chars_per_result": self.max_chars_per_extract
                }
            elif isinstance(full_content, FullContentSettings):
                full_content_param = full_content.model_dump(exclude_none=True)

            # Build excerpts config
            excerpts_param = excerpts
            if isinstance(excerpts, ExcerptSettings):
                excerpts_param = excerpts.model_dump(exclude_none=True)

            # Build fetch_policy config
            fetch_policy_param = None
            if fetch_policy:
                fetch_policy_param = fetch_policy.model_dump(exclude_none=True)

            # Extract content from URLs
            extract_response = await client.extract(
                urls=urls,
                objective=search_objective,
                search_queries=search_queries,
                excerpts=excerpts_param,
                full_content=full_content_param,
                fetch_policy=fetch_policy_param,
            )

            results = extract_response.get("results", [])
            errors = extract_response.get("errors", [])

            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "url": result.get("url"),
                    "title": result.get("title"),
                }

                # Add excerpts if present
                if "excerpts" in result and result["excerpts"] is not None:
                    formatted_result["excerpts"] = result["excerpts"]
                    # Combine excerpts into content field for backward compatibility
                    if "full_content" not in result:
                        # Excerpts are a list of strings, join them with newlines
                        formatted_result["content"] = "\n\n".join(result["excerpts"])

                # Add full_content if present
                if "full_content" in result:
                    formatted_result["full_content"] = result["full_content"]
                    # For backward compatibility, also set as "content"
                    formatted_result["content"] = result["full_content"]

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
                        "content": f"Error: {error.get('error_type', 'Unknown error')}",
                        "error_type": error.get("error_type"),
                        "http_status_code": error.get("http_status_code"),
                    }
                    for error in errors
                ]
            )

            return formatted_results

        except Exception as e:
            msg = f"Error calling Parallel Extract API: {e!s}"
            raise ValueError(msg) from e
