"""Anvesha - Research Assistant Kit for ML and CV research.

A phase-driven research SDK: each of the eleven phases runs as a discrete
CLI command and produces a structured Markdown file in the researcher's
project workspace. See ANVESHA.md for the project charter.
"""

from anvesha.core.adapters import create_llm_client
from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient, MCPToolDef
from anvesha.core.config import PipelineConfig, load_config

__version__ = "0.1.0"

__all__ = [
    "LLMClient",
    "LLMResponse",
    "MCPClient",
    "MCPToolDef",
    "PipelineConfig",
    "create_llm_client",
    "load_config",
    "__version__",
]
