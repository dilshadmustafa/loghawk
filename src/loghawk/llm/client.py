"""Provider-neutral chat and embedding calls backed by the LiteLLM SDK."""

from collections.abc import Iterator, Sequence
import os
from typing import Any

import loghawk.config as config


def _chat_model(model: str | None = None) -> str:
    name = (model or config.LH_LLM_MODEL).strip()
    if "/" in name:
        return name
    prefixes = {
        "ollama": "ollama_chat/",
        "bedrock": "bedrock/",
        "azure_ai": "azure_ai/",
    }
    return prefixes[config.LH_LLM_PROVIDER] + name


def _embedding_model(model: str | None = None) -> str:
    name = (model or config.LH_EMBEDDING_MODEL).strip()
    if "/" in name:
        return name
    prefixes = {
        "ollama": "ollama/",
        "bedrock": "bedrock/",
        "azure_ai": "azure_ai/",
    }
    return prefixes[config.LH_LLM_PROVIDER] + name


class LiteLLMClient:
    """Small LogHawk interface over LiteLLM's normalized SDK responses."""

    def _chat_kwargs(self, model: str | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": _chat_model(model),
            "timeout": config.LH_LLM_TIMEOUT_SECONDS,
        }
        if config.LH_LLM_BASE_URL:
            kwargs["api_base"] = config.LH_LLM_BASE_URL
        if config.LH_LLM_API_KEY:
            kwargs["api_key"] = config.LH_LLM_API_KEY
        if config.LH_LLM_PROVIDER == "bedrock":
            region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            if region:
                kwargs["aws_region_name"] = region
        return kwargs

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        response_schema: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        content, _response = self.complete_with_response(
            messages,
            response_schema=response_schema,
            model=model,
            temperature=temperature,
        )
        return content

    def complete_with_response(
        self,
        messages: Sequence[dict[str, str]],
        *,
        response_schema: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> tuple[str, Any]:
        from litellm import completion

        kwargs = self._chat_kwargs(model)
        kwargs.update(messages=list(messages), temperature=temperature)
        if response_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "loghawk_structured_response",
                    "schema": response_schema,
                },
            }
        response = completion(**kwargs)
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if not content:
            raise RuntimeError(
                f"LLM provider returned an empty response (model={kwargs['model']})."
            )
        return str(content), response

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        from litellm import completion

        kwargs = self._chat_kwargs(model)
        kwargs.update(messages=list(messages), temperature=temperature, stream=True)
        for chunk in completion(**kwargs):
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            content = getattr(choices[0].delta, "content", None)
            if content:
                yield str(content)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        from litellm import embedding

        kwargs: dict[str, Any] = {
            "model": _embedding_model(),
            "input": list(texts),
            "timeout": config.LH_LLM_TIMEOUT_SECONDS,
        }
        if config.LH_EMBEDDING_BASE_URL:
            kwargs["api_base"] = config.LH_EMBEDDING_BASE_URL
        if config.LH_EMBEDDING_API_KEY:
            kwargs["api_key"] = config.LH_EMBEDDING_API_KEY
        response = embedding(**kwargs)
        def item_index(item: Any) -> int:
            return item["index"] if isinstance(item, dict) else item.index

        def item_embedding(item: Any) -> list[float]:
            value = item["embedding"] if isinstance(item, dict) else item.embedding
            return list(value)

        items = sorted(response.data, key=item_index)
        vectors = [item_embedding(item) for item in items]
        if len(vectors) != len(texts):
            raise RuntimeError(
                "Embedding provider returned a different number of vectors "
                f"than inputs ({len(vectors)} != {len(texts)})."
            )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise RuntimeError("Embedding provider returned inconsistent vector dimensions.")
        return vectors


def get_llm_client() -> LiteLLMClient:
    return LiteLLMClient()
