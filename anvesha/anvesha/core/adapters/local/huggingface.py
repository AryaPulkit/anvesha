"""In-process HuggingFace transformers adapter (ANVESHA.md S10.3).

No server: the model is loaded into this process. transformers/torch are
optional dependencies, lazily imported inside :meth:`load` so the rest of
the SDK works without them. ``reported_vram_gb`` is auto-detected at load
time from the model's parameter bytes (S13.5).

v0 limitation: this backend has no tool-call loop, so passing ``mcps`` to
:meth:`run` raises :class:`AdapterError`. Use a served backend (vLLM,
Ollama, llama.cpp) for phases that need MCP tools.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient
from anvesha.core.exceptions import AdapterError

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "the HuggingFace backend needs the optional 'transformers' and 'torch' "
    'dependencies - install them with: pip install "anvesha[hf]"'
)


class HuggingFaceAdapterConfig(BaseModel):
    """Configuration for the in-process HuggingFace backend."""

    model_name: str
    device: str = "auto"
    max_new_tokens: int = 4096


class HuggingFaceAdapter(LLMClient):
    """LLMClient running a transformers model in-process (no server)."""

    def __init__(self, config: HuggingFaceAdapterConfig):
        self.config = config
        self.name = config.model_name
        self.reported_vram_gb: float | None = None
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        """Load tokenizer and model; auto-detect VRAM from parameter bytes."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise AdapterError(f"{self.name}: {_INSTALL_HINT}") from exc

        logger.info("loading HuggingFace model %s", self.config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, device_map=self.config.device
        )
        param_bytes = sum(
            p.numel() * p.element_size() for p in self._model.parameters()
        )
        self.reported_vram_gb = param_bytes / 1024**3
        logger.info(
            "%s loaded (reported_vram_gb=%.1f)", self.name, self.reported_vram_gb
        )

    def unload(self) -> None:
        """Drop the model and empty the CUDA cache when torch is present."""
        self._model = None
        self._tokenizer = None
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(
        self,
        prompt: str,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> LLMResponse:
        if mcps:
            raise AdapterError(
                f"{self.name}: the in-process HuggingFace backend does not "
                "support MCP tool calls in v0 - use a served backend instead"
            )
        if self._model is None:
            self.load()

        full_prompt = self.inject_input_files(prompt, input_files)
        if getattr(self._tokenizer, "chat_template", None):
            text = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": full_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = full_prompt

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(
            **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False
        )
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][prompt_len:]
        content = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return LLMResponse(
            content=content,
            tool_calls_made=0,
            tokens_in=int(prompt_len),
            tokens_out=int(len(new_ids)),
        )
