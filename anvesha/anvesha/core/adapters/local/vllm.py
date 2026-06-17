"""vLLM adapter (GPT-OSS-120B and general vLLM) - S15.1.

vLLM speaks the OpenAI chat-completions API, so almost everything comes
from :class:`OpenAICompatAdapter`. The two vLLM-specific behaviours:

- Harmony format for GPT-OSS models: a ``"Reasoning: {effort}"`` system
  prompt header (standard ChatML makes GPT-OSS models behave incorrectly).
- Native structured output via vLLM's ``guided_json`` request parameter,
  with a fallback to the prompt-based default when the server rejects it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Sequence

import openai

from anvesha.core.adapters.base import MCPClient, parse_json_response
from anvesha.core.adapters.openai_compat import OpenAICompatAdapter, OpenAICompatConfig
from anvesha.core.exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterTimeoutError,
)

logger = logging.getLogger(__name__)


class VLLMAdapterConfig(OpenAICompatConfig):
    """Configuration for a vLLM OpenAI-compatible server (S15.1)."""

    base_url: str | None = "http://localhost:8000/v1"
    #: Set true for GPT-OSS-120B / GPT-OSS-20B (Harmony chat template).
    harmony_format: bool = False
    #: GPT-OSS reasoning effort; applied via Harmony format only.
    reasoning_effort: Literal["low", "medium", "high"] | None = "high"
    #: Reference value for GPT-OSS-120B (S13.5); the GPU allocator reads this.
    reported_vram_gb: float | None = 40.0


class VLLMAdapter(OpenAICompatAdapter):
    """LLMClient over a vLLM server."""

    config_cls = VLLMAdapterConfig
    config: VLLMAdapterConfig

    def system_prompt_header(self) -> str | None:
        if self.config.harmony_format and self.config.reasoning_effort:
            return f"Reasoning: {self.config.reasoning_effort}"
        return None

    def structured_run(
        self,
        prompt: str,
        json_schema: dict,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> dict:
        """Structured output via vLLM ``guided_json`` (single chat call).

        ``guided_json`` constrains a single completion and cannot drive the
        tool-call loop, so when ``mcps`` are supplied we use the prompt-based
        base implementation instead. If the server rejects ``guided_json``
        (openai.BadRequestError), we also fall back to the base behaviour.
        """
        if mcps:
            return super().structured_run(
                prompt, json_schema, mcps=mcps, input_files=input_files
            )

        full_prompt = self.inject_input_files(prompt, input_files)
        messages: list[dict[str, Any]] = []
        header = self.system_prompt_header()
        if header:
            messages.append({"role": "system", "content": header})
        messages.append({"role": "user", "content": full_prompt})

        request: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "extra_body": {"guided_json": json_schema},
        }
        if self.config.max_tokens is not None:
            request["max_tokens"] = self.config.max_tokens

        try:
            response = self.client.chat.completions.create(**request)
        except openai.BadRequestError as exc:
            logger.info(
                "%s: server rejected guided_json (%s); falling back to "
                "prompt-based structured output",
                self.name,
                exc,
            )
            return super().structured_run(
                prompt, json_schema, mcps=mcps, input_files=input_files
            )
        except openai.APITimeoutError as exc:
            raise AdapterTimeoutError(
                f"{self.name}: request timed out after "
                f"{self.config.timeout_seconds}s"
            ) from exc
        except openai.APIConnectionError as exc:
            raise AdapterConnectionError(
                f"{self.name}: cannot reach backend at "
                f"{self.config.base_url}: {exc}"
            ) from exc
        except openai.OpenAIError as exc:
            raise AdapterError(f"{self.name}: backend error: {exc}") from exc

        return parse_json_response(response.choices[0].message.content or "")
