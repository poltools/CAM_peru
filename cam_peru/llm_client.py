"""
Thin wrapper around Azure OpenAI used by every classification/extraction
script in this package.

Credentials are ALWAYS read from environment variables — never hardcoded.
See ``.env.example`` for the expected variable names. A ``.env`` file in the
repo root is auto-loaded if :mod:`python_dotenv` is installed.
"""

from __future__ import annotations

import os
from typing import Optional

try:  # Optional .env loading — degrades gracefully if the package is missing.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from openai import AzureOpenAI

from .config import AZURE_ENV_VARS, DEFAULT_API_VERSION, DEFAULT_DEPLOYMENT


def get_azure_client(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
) -> AzureOpenAI:
    """Construct an authenticated :class:`AzureOpenAI` client.

    Raises
    ------
    RuntimeError
        If no endpoint or API key is resolvable from the environment.
    """
    endpoint = endpoint or os.getenv(AZURE_ENV_VARS["endpoint"])
    api_key = api_key or os.getenv(AZURE_ENV_VARS["api_key"])
    api_version = api_version or os.getenv(AZURE_ENV_VARS["api_version"], DEFAULT_API_VERSION)

    if not endpoint or not api_key:
        raise RuntimeError(
            "Azure OpenAI credentials are not set. Populate a .env file "
            "(see .env.example) or export ENDPOINT_URL and "
            "AZURE_OPENAI_API_KEY in your shell before running."
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def get_deployment_name() -> str:
    """Return the configured deployment name (defaults to gpt-4o)."""
    return os.getenv(AZURE_ENV_VARS["deployment"], DEFAULT_DEPLOYMENT)


def chat_completion(
    client: AzureOpenAI,
    prompt: str,
    model: Optional[str] = None,
    system_prompt: str = (
        "You are a classifier that maps text to predefined categories "
        "based on detailed descriptions."
    ),
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = 42,
) -> str:
    """Issue a single chat-completion request and return the stripped text."""
    model = model or get_deployment_name()
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if seed is not None:
        kwargs["seed"] = seed

    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content.strip()
