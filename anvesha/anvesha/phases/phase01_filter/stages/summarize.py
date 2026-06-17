"""Stage 9 -- Summary Generation (LLM), spec S5 Stage9 summary rubric.

For each retained paper the LLM writes a concise, implementation-focused
summary covering, in order: problem addressed; methodology; key contributions;
implementation relevance; practical usefulness (target 120-200 words, no
marketing language, no claims unsupported by title/abstract). Greedy decoding
for determinism (the injected adapter defaults to temperature 0.0).

Summary generation is non-fatal (EH-6): after ``summary_max_retries`` failed
attempts the body becomes ``"Summary generation failed."`` and
``summary_status = "failed"``; the paper and its metadata are kept.
"""

from __future__ import annotations

import logging

from anvesha.phases.phase01_filter.schemas.candidate import RetainedPaper
from anvesha.phases.phase01_filter.schemas.config import Options
from anvesha.core.adapters.base import LLMClient

logger = logging.getLogger(__name__)

# EH-6 fallback body emitted into the record when generation fails.
FAILED_SUMMARY = "Summary generation failed."

_PROMPT_TEMPLATE = (
    "You are writing an implementation-focused summary of a single research "
    "paper for an engineering audience. Use ONLY the title and abstract below; "
    "do not invent details, and do not use marketing language.\n\n"
    "Title: {title}\n"
    "Abstract: {abstract}\n\n"
    "Write one cohesive summary of 120-200 words that covers, in this order:\n"
    "1. The problem the paper addresses.\n"
    "2. The methodology.\n"
    "3. The key contributions.\n"
    "4. Implementation relevance (what an engineer would build or reuse).\n"
    "5. Practical usefulness.\n\n"
    "Make no claims that are not supported by the title or abstract. "
    "Respond with the summary text only -- no headings, no preamble."
)

_NO_ABSTRACT = "(no abstract available)"


def _build_prompt(title: str, abstract: str | None) -> str:
    """Render the summary prompt for one paper (S5 Stage9 rubric)."""
    abstract_text = abstract.strip() if abstract and abstract.strip() else _NO_ABSTRACT
    return _PROMPT_TEMPLATE.format(title=title.strip(), abstract=abstract_text)


def summarize_paper(
    llm: LLMClient,
    paper: RetainedPaper,
    abstract: str | None,
    options: Options,
) -> None:
    """Generate the implementation-focused summary for one paper (S5 Stage9).

    Calls ``llm.run`` with a greedy prompt up to ``options.summary_max_retries``
    times (at least once). On the first attempt that returns non-empty text,
    sets ``paper.summary`` to the stripped text and ``paper.summary_status =
    "ok"``. If every attempt fails or returns empty, sets ``paper.summary =
    "Summary generation failed."`` and ``paper.summary_status = "failed"``
    (EH-6, non-fatal). Mutates ``paper`` in place; never raises.
    """
    prompt = _build_prompt(paper.title, abstract)
    attempts = max(1, options.summary_max_retries)
    summary: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = llm.run(prompt)
        except Exception as exc:  # noqa: BLE001 - non-fatal per paper (EH-6)
            logger.warning(
                "Summary attempt %d/%d for %s raised: %s",
                attempt,
                attempts,
                paper.paper_id,
                exc,
            )
            continue
        text = response.content.strip()
        if text:
            summary = text
            break
        logger.info(
            "Summary attempt %d/%d for %s returned empty text",
            attempt,
            attempts,
            paper.paper_id,
        )

    if summary:
        paper.summary = summary
        paper.summary_status = "ok"
    else:
        paper.summary = FAILED_SUMMARY
        paper.summary_status = "failed"
