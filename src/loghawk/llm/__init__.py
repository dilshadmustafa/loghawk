"""LogHawk provider-neutral LLM and embedding clients."""

from .client import LiteLLMClient, get_llm_client

__all__ = ["LiteLLMClient", "get_llm_client"]
