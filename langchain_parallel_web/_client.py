"""Client utilities for Parallel AI integration."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import openai
from parallel import AsyncParallel, Parallel


def get_api_key(api_key: Optional[str] = None) -> str:
    """Retrieve the Parallel AI API key from argument or environment variables.

    Args:
        api_key: Optional API key string.

    Returns:
        API key string.

    Raises:
        ValueError: If API key is not found.
    """
    if api_key:
        return api_key

    env_key = os.environ.get("PARALLEL_AI_API_KEY")
    if env_key:
        return env_key

    msg = (
        "Parallel AI API key not found. Please pass it as an argument or set the "
        "PARALLEL_AI_API_KEY environment variable."
    )
    raise ValueError(msg)


def get_openai_client(api_key: str, base_url: str) -> openai.OpenAI:
    """Returns a configured sync OpenAI client for the Chat API."""
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def get_async_openai_client(api_key: str, base_url: str) -> openai.AsyncOpenAI:
    """Returns a configured async OpenAI client for the Chat API."""
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)


class ParallelSearchClient:
    """Synchronous client for Parallel AI Search API using the Parallel SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.parallel.ai",
        environment: str = "production",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.environment = environment
        # Initialize the Parallel SDK client
        self.client = Parallel(api_key=api_key, base_url=base_url)

    def search(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        processor: str = "base",
        max_results: int = 10,
        max_chars_per_result: int = 1500,
        source_policy: Optional[dict[str, Union[str, list[str]]]] = None,
    ) -> dict[str, Any]:
        """Perform a synchronous search using the Parallel AI Search API via SDK."""
        if not objective and not search_queries:
            msg = "Either 'objective' or 'search_queries' must be provided"
            raise ValueError(msg)

        # Set timeout based on processor type:
        # - base processor: 10 seconds (typical 4-5s response time)
        # - pro processor: 90 seconds (typical 45-70s response time)
        timeout = 90.0 if processor == "pro" else 10.0

        # Use the Parallel SDK's beta.search method
        search_response = self.client.beta.search(
            objective=objective,
            search_queries=search_queries,
            processor=processor,
            max_results=max_results,
            max_chars_per_result=max_chars_per_result,
            source_policy=source_policy,
            timeout=timeout,
        )

        # Convert the SDK response to a dictionary
        return search_response.model_dump()


class AsyncParallelSearchClient:
    """Asynchronous client for Parallel AI Search API using the Parallel SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.parallel.ai",
        environment: str = "production",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.environment = environment
        # Initialize the Parallel SDK async client
        self.client = AsyncParallel(api_key=api_key, base_url=base_url)

    async def search(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        processor: str = "base",
        max_results: int = 10,
        max_chars_per_result: int = 1500,
        source_policy: Optional[dict[str, Union[str, list[str]]]] = None,
    ) -> dict[str, Any]:
        """Perform an async search using the Parallel AI Search API via SDK."""
        if not objective and not search_queries:
            msg = "Either 'objective' or 'search_queries' must be provided"
            raise ValueError(msg)

        # Set timeout based on processor type:
        # - base processor: 10 seconds (typical 4-5s response time)
        # - pro processor: 90 seconds (typical 45-70s response time)
        timeout = 90.0 if processor == "pro" else 10.0

        # Use the Parallel SDK's beta.search method
        search_response = await self.client.beta.search(
            objective=objective,
            search_queries=search_queries,
            processor=processor,
            max_results=max_results,
            max_chars_per_result=max_chars_per_result,
            source_policy=source_policy,
            timeout=timeout,
        )

        # Convert the SDK response to a dictionary
        return search_response.model_dump()


def get_search_client(
    api_key: str, base_url: str = "https://api.parallel.ai"
) -> ParallelSearchClient:
    """Returns a configured sync Parallel AI Search client."""
    return ParallelSearchClient(api_key, base_url)


def get_async_search_client(
    api_key: str, base_url: str = "https://api.parallel.ai"
) -> AsyncParallelSearchClient:
    """Returns a configured async Parallel AI Search client."""
    return AsyncParallelSearchClient(api_key, base_url)
