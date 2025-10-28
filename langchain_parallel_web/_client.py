"""Client utilities for Parallel integration."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import openai
from parallel import AsyncParallel, Parallel


def get_api_key(api_key: Optional[str] = None) -> str:
    """Retrieve the Parallel API key from argument or environment variables.

    Args:
        api_key: Optional API key string.

    Returns:
        API key string.

    Raises:
        ValueError: If API key is not found.
    """
    if api_key:
        return api_key

    env_key = os.environ.get("PARALLEL_API_KEY")
    if env_key:
        return env_key

    msg = (
        "Parallel API key not found. Please pass it as an argument or set the "
        "PARALLEL_API_KEY environment variable."
    )
    raise ValueError(msg)


def get_openai_client(api_key: str, base_url: str) -> openai.OpenAI:
    """Returns a configured sync OpenAI client for the Chat API."""
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def get_async_openai_client(api_key: str, base_url: str) -> openai.AsyncOpenAI:
    """Returns a configured async OpenAI client for the Chat API."""
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

def _get_search_timeout(processor: str) -> float:
    """Get the timeout for the search API based on the processor type."""
    # Set timeout based on processor type:
    # - base processor: 10 seconds (typical 4-5s response time)
    # - pro processor: 90 seconds (typical 45-70s response time)
    return 90.0 if processor == "pro" else 10.0

class ParallelSearchClient:
    """Synchronous client for Parallel Search API using the Parallel SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.parallel.ai",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
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
        """Perform a synchronous search using the Parallel Search API via SDK."""
        if not objective and not search_queries:
            msg = "Either 'objective' or 'search_queries' must be provided"
            raise ValueError(msg)


        # Use the Parallel SDK's beta.search method
        search_response = self.client.beta.search(
            objective=objective,
            search_queries=search_queries,
            processor=processor,
            max_results=max_results,
            max_chars_per_result=max_chars_per_result,
            source_policy=source_policy,
            timeout=_get_search_timeout(processor),
        )

        # Convert the SDK response to a dictionary
        return search_response.model_dump()


class AsyncParallelSearchClient:
    """Asynchronous client for Parallel Search API using the Parallel SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.parallel.ai",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
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
        """Perform an async search using the Parallel Search API via SDK."""
        if not objective and not search_queries:
            msg = "Either 'objective' or 'search_queries' must be provided"
            raise ValueError(msg)

        # Use the Parallel SDK's beta.search method
        search_response = await self.client.beta.search(
            objective=objective,
            search_queries=search_queries,
            processor=processor,
            max_results=max_results,
            max_chars_per_result=max_chars_per_result,
            source_policy=source_policy,
            timeout=_get_search_timeout(processor),
        )

        # Convert the SDK response to a dictionary
        return search_response.model_dump()


def get_search_client(
    api_key: str, base_url: str = "https://api.parallel.ai"
) -> ParallelSearchClient:
    """Returns a configured sync Parallel Search client."""
    return ParallelSearchClient(api_key, base_url)


def get_async_search_client(
    api_key: str, base_url: str = "https://api.parallel.ai"
) -> AsyncParallelSearchClient:
    """Returns a configured async Parallel Search client."""
    return AsyncParallelSearchClient(api_key, base_url)


class ParallelExtractClient:
    """Synchronous client for Parallel Extract API using the Parallel SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.parallel.ai",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # Initialize the Parallel SDK client
        self.client = Parallel(api_key=api_key, base_url=base_url)

    def extract(
        self,
        urls: list[str],
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        excerpts: Optional[Union[bool, dict[str, Any]]] = None,
        full_content: Optional[Union[bool, dict[str, Any]]] = None,
        fetch_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Perform a synchronous extract using the Parallel Extract API via SDK."""
        if not urls:
            msg = "At least one URL must be provided"
            raise ValueError(msg)

        # Set timeout based on number of URLs (5 seconds per URL)
        timeout = 5.0 * len(urls)

        # Use the Parallel SDK's beta.extract method
        extract_response = self.client.beta.extract(
            urls=urls,
            objective=objective,
            search_queries=search_queries,
            excerpts=excerpts,
            full_content=full_content,
            fetch_policy=fetch_policy,
            betas=["search-extract-2025-10-10"],
            timeout=timeout,
        )

        # Convert the SDK response to a dictionary
        return extract_response.model_dump()


class AsyncParallelExtractClient:
    """Asynchronous client for Parallel Extract API using the Parallel SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.parallel.ai",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # Initialize the Parallel SDK async client
        self.client = AsyncParallel(api_key=api_key, base_url=base_url)

    async def extract(
        self,
        urls: list[str],
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        excerpts: Optional[Union[bool, dict[str, Any]]] = None,
        full_content: Optional[Union[bool, dict[str, Any]]] = None,
        fetch_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Perform an async extract using the Parallel Extract API via SDK."""
        if not urls:
            msg = "At least one URL must be provided"
            raise ValueError(msg)

        # Set timeout based on number of URLs (5 seconds per URL)
        timeout = 5.0 * len(urls)

        # Use the Parallel SDK's beta.extract method
        extract_response = await self.client.beta.extract(
            urls=urls,
            objective=objective,
            search_queries=search_queries,
            excerpts=excerpts,
            full_content=full_content,
            fetch_policy=fetch_policy,
            betas=["search-extract-2025-10-10"],
            timeout=timeout,
        )

        # Convert the SDK response to a dictionary
        return extract_response.model_dump()


def get_extract_client(
    api_key: str, base_url: str = "https://api.parallel.ai"
) -> ParallelExtractClient:
    """Returns a configured sync Parallel Extract client."""
    return ParallelExtractClient(api_key, base_url)


def get_async_extract_client(
    api_key: str, base_url: str = "https://api.parallel.ai"
) -> AsyncParallelExtractClient:
    """Returns a configured async Parallel Extract client."""
    return AsyncParallelExtractClient(api_key, base_url)
