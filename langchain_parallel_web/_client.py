"""Client utilities for Parallel AI integration."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import httpx
import openai


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
    """Synchronous client for Parallel AI Search API."""

    def __init__(self, api_key: str, base_url: str = "https://api.parallel.ai"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.search_url = f"{self.base_url}/v1beta/search"

    def search(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        processor: str = "base",
        max_results: int = 10,
        max_chars_per_result: int = 1500,
        source_policy: Optional[dict[str, Union[str, list[str]]]] = None,
    ) -> dict[str, Any]:
        """Perform a synchronous search using the Parallel AI Search API."""
        if not objective and not search_queries:
            msg = "Either 'objective' or 'search_queries' must be provided"
            raise ValueError(msg)

        payload = {
            "processor": processor,
            "max_results": max_results,
            "max_chars_per_result": max_chars_per_result,
        }

        if objective:
            payload["objective"] = objective
        if search_queries:
            payload["search_queries"] = search_queries
        if source_policy:
            payload["source_policy"] = source_policy

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

        # Set timeout based on processor type:
        # - base processor: 10 seconds (typical 4-5s response time)
        # - pro processor: 90 seconds (typical 45-70s response time)
        timeout = 90.0 if processor == "pro" else 10.0

        with httpx.Client(timeout=timeout) as client:
            response = client.post(self.search_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()


def get_search_client(
    api_key: str, base_url: str = "https://api.parallel.ai"
) -> ParallelSearchClient:
    """Returns a configured sync Parallel AI Search client."""
    return ParallelSearchClient(api_key, base_url)
