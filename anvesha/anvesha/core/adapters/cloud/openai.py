"""OpenAI cloud adapter (ANVESHA.md S10.2).

The OpenAI platform speaks the same chat-completions protocol as the local
backends, so this adapter is a thin subclass of OpenAICompatAdapter: it only
changes how the API key is resolved and defaults ``base_url`` to the
platform endpoint.
"""

from __future__ import annotations

import logging
import os

from anvesha.core.adapters.openai_compat import OpenAICompatAdapter, OpenAICompatConfig
from anvesha.core.exceptions import AdapterError

logger = logging.getLogger(__name__)


class CloudOpenAIConfig(OpenAICompatConfig):
    """Configuration for :class:`OpenAICloudAdapter`.

    ``base_url=None`` uses the OpenAI platform default endpoint.
    ``api_key=None`` falls back to the ``OPENAI_API_KEY`` environment
    variable at client build time.
    """

    base_url: str | None = None
    api_key: str | None = None


class OpenAICloudAdapter(OpenAICompatAdapter):
    """LLMClient over the OpenAI cloud API."""

    config_cls: type[OpenAICompatConfig] = CloudOpenAIConfig

    def _resolve_api_key(self) -> str:
        key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise AdapterError(
                f"{self.name}: no OpenAI API key - set api_key in the "
                "backend config or the OPENAI_API_KEY environment variable"
            )
        return key
