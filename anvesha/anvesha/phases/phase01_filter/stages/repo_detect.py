"""Stage 7 -- Repository Detection (deterministic, metadata-only), spec S8.

For each retained paper we set ``code_url`` and ``git_exists`` from the
*metadata already gathered by the discovery tools only*. No GitHub API calls,
no web searches, no repository discovery of any kind (FR-7 / S8 opening). A
code link qualifies only if it is present in the candidate's metadata.

The stage operates per retained paper; the orchestrator pairs each
:class:`RetainedPaper` with the :class:`PaperCandidate` it was built from.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from anvesha.phases.phase01_filter.schemas.candidate import (
    PaperCandidate,
    RetainedPaper,
)

logger = logging.getLogger(__name__)

# Known code-hosting domains (S8.1 step 2). Matched case-insensitively against
# the URL host, including ``www.`` and other leading subdomains
# (e.g. ``raw.githubusercontent.com`` -> matched via github host suffix is NOT
# included; only the registrable hosts the spec lists). The set is fixed here;
# making it configurable is a S13 extensibility item, out of scope for v2.0.
CODE_HOSTS: tuple[str, ...] = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "huggingface.co",
    "codeberg.org",
)


def _host_is_code_host(url: str) -> bool:
    """True if ``url`` parses to a host on a known code-hosting domain.

    Tolerant of malformed input (EH-10): any parsing problem yields ``False``
    rather than raising. The host is matched as the registrable domain or any
    subdomain of it (so ``www.github.com`` qualifies).
    """
    try:
        parts = urlsplit(url.strip())
    except (ValueError, AttributeError):
        # Malformed URL or non-string: ignore the link (EH-10).
        return False
    host = (parts.hostname or "").lower()
    if not host:
        return False
    for code_host in CODE_HOSTS:
        if host == code_host or host.endswith("." + code_host):
            return True
    return False


def _is_well_formed(url: str) -> bool:
    """True if ``url`` has a scheme and a host (rejects malformed links, EH-10)."""
    try:
        parts = urlsplit(url.strip())
    except (ValueError, AttributeError):
        return False
    return bool(parts.scheme and parts.hostname)


def detect_repository(paper: RetainedPaper, candidate: PaperCandidate) -> None:
    """Set ``paper.code_url`` / ``paper.git_exists`` from metadata only (S8.2).

    Detection procedure (S8.1):

    1. Prefer an explicit code-link metadata field
       (``candidate.metadata_code_url``) when it is a well-formed URL.
    2. Otherwise scan the candidate's other URL fields (``paper_url``) for a
       link whose host is a known code-hosting domain (:data:`CODE_HOSTS`).

    Field-setting rules (S8.2): a found link sets ``git_exists = True`` and
    ``code_url = <link>``; otherwise ``git_exists = False`` and
    ``code_url = None``. The S8.3 invariant ``git_exists == (code_url != null)``
    is enforced before returning. Malformed URLs are ignored, never raised on
    (EH-10). Mutates ``paper`` in place; returns ``None``.
    """
    code_url: str | None = None

    # Step 1: explicit code/repository metadata field. It is already the
    # selected link per S7.3 rule 3 (the dedup merge picked it); we accept it
    # iff it is a well-formed URL. A malformed explicit link is dropped (EH-10)
    # and we fall through to scanning other URL fields.
    explicit = candidate.metadata_code_url
    if explicit and _is_well_formed(explicit):
        code_url = explicit.strip()
    elif explicit:
        logger.warning(
            "Ignoring malformed explicit code link %r for %s (EH-10)",
            explicit,
            paper.paper_id,
        )

    # Step 2: scan other URL fields for a known code-host link. Only
    # ``paper_url`` is a URL field on the candidate; checked in a fixed order
    # so detection is deterministic if more fields are added later.
    if code_url is None:
        for url in (candidate.paper_url,):
            if url and _host_is_code_host(url):
                code_url = url.strip()
                break

    paper.code_url = code_url
    paper.git_exists = code_url is not None

    # S8.3 invariant: never let git_exists diverge from code_url presence.
    assert paper.git_exists == (paper.code_url is not None)
