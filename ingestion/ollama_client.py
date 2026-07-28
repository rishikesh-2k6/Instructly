"""Thin wrapper around the local Ollama server.

Two entry points:
  - generate_json: schema-constrained chat generation, with validation and
    retry, since local models are less reliable at strict structured
    output than hosted frontier models.
  - embed: embedding generation.

Ollama's `format` chat parameter accepts a JSON schema dict directly
(verified against the installed `ollama` client — see its `Client.chat`
signature), which requests grammar-constrained output from the server.
That alone is not a guarantee on local models, so every response is still
parsed and validated against a pydantic schema regardless.
"""

from __future__ import annotations

import json
import logging
from typing import Type, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 2  # total attempts = 1 initial + MAX_RETRIES retries

_client = ollama.Client(host=config.OLLAMA_HOST)


class GenerationError(RuntimeError):
    """Raised when the model fails to produce schema-valid JSON after retries."""


def _strict_retry_prompt(original_prompt: str, json_schema: dict, error: Exception) -> str:
    return (
        f"{original_prompt}\n\n"
        f"Your previous response was invalid ({error}). "
        "Return ONLY valid JSON matching this exact schema, with no other text, "
        "no markdown code fences, and no explanation before or after the JSON:\n"
        f"{json.dumps(json_schema)}"
    )


def generate_json(prompt: str, schema: Type[T], model: str | None = None) -> T:
    """Call the chat model and parse+validate its response against `schema`.

    Retries up to MAX_RETRIES times with a stricter prompt if the response
    isn't valid JSON matching `schema`. Raises GenerationError if still
    invalid after retries — callers decide how to handle the failure.
    """
    model = model or config.OLLAMA_MODEL
    json_schema = schema.model_json_schema()
    current_prompt = prompt
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        response = _client.chat(
            model=model,
            messages=[{"role": "user", "content": current_prompt}],
            format=json_schema,
            options={"temperature": 0},
        )
        raw = response.message.content
        try:
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "generate_json attempt %d/%d failed schema %s: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                schema.__name__,
                exc,
            )
            current_prompt = _strict_retry_prompt(prompt, json_schema, exc)

    raise GenerationError(
        f"Model '{model}' failed to produce JSON matching schema {schema.__name__} "
        f"after {MAX_RETRIES + 1} attempts. Last error: {last_error}"
    )


def embed(text: str, model: str | None = None) -> list[float]:
    """Generate an embedding vector for `text` via the local embedding model."""
    model = model or config.OLLAMA_EMBED_MODEL
    response = _client.embed(model=model, input=text)
    return response.embeddings[0]
