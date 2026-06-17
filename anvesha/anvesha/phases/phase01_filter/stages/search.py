"""Stage 2 - Multi-Source Search (MCP), spec S5 Stage 2 / S2.

Runs every Stage-1 query against the two mandated discovery sources -- PapersFlow
and paper-search-mcp -- via the injected :class:`ToolRouter`, normalizes each raw
result into a :class:`PaperCandidate`, and tags it with its ``source`` and the
query that surfaced it. ``identified`` (the funnel's first count) is the total raw
candidate count returned here.

Tool calls go directly through the router (NOT an LLM tool loop). Sources are
optional individually: if one source's tools are absent we skip it and degrade
coverage (EH-2). If NO search tool is available at all we raise (EH-3 -> fatal at
the orchestrator).

Determinism (NFR-1): queries are processed in their given order; for each query,
sources are tried in a fixed order; within a source, results preserve the tool's
returned order. No concurrency is used (the spec permits sequential execution and
it keeps ordering reproducible).
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

from anvesha.core.exceptions import AnveshaError
from anvesha.phases.phase01_filter.schemas.candidate import PaperCandidate
from anvesha.phases.phase01_filter.tools import ToolRouter

logger = logging.getLogger(__name__)

# Discovery tool names per source, in the order Stage 2 tries them. The first
# tool a source advertises is used; if none are present that source is skipped.
# Names are namespaced (server.tool) per the MCP adapter contract.
_PAPERSFLOW_TOOLS: tuple[str, ...] = ("papersflow.search", "papersflow.search_literature")
_PAPER_SEARCH_TOOLS: tuple[str, ...] = (
    "paper-search.search_papers",
    "paper-search.search_arxiv",
)

# (logical source label emitted on the candidate, ordered tool-name candidates).
# Order here fixes the source-iteration order for determinism.
_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("papersflow", _PAPERSFLOW_TOOLS),
    ("paper_search", _PAPER_SEARCH_TOOLS),
)

# Code-hosting domains whose URLs count as an explicit code link in metadata
# (spec S8.1). Used by normalization to seed ``metadata_code_url`` from a generic
# URL field when no dedicated code field is present. Stage 7 owns the final
# selection; here we only surface a plausible link.
_CODE_HOST_SUFFIXES: tuple[str, ...] = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "huggingface.co",
    "codeberg.org",
)

# Field-name aliases tolerated across the two sources' result schemas (EH-4: any
# missing field maps to None). The first present, non-empty alias wins.
_TITLE_KEYS = ("title", "name")
_ABSTRACT_KEYS = ("abstract", "summary", "description")
_VENUE_KEYS = ("venue", "venue_raw", "journal", "publication_venue", "container", "booktitle")
_YEAR_KEYS = ("year", "publication_year", "pub_year")
_CITATION_KEYS = ("citation_count", "citationCount", "citations", "n_citations")
_DOI_KEYS = ("doi", "DOI")
_ARXIV_KEYS = ("arxiv_id", "arxivId", "arxiv", "arXiv")
_URL_KEYS = ("paper_url", "url", "pdf_url", "landing_page", "link", "openAccessPdf")
_CODE_KEYS = ("code", "code_url", "repository", "repo", "repo_url", "github", "pwc_url")
_LANG_KEYS = ("language", "lang")
_AUTHOR_KEYS = ("authors", "author", "authorships")


def _first(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first present, non-None value among ``keys`` in ``d``."""
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _as_str(value: Any) -> str | None:
    """Coerce a scalar to a trimmed non-empty string, else None."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    # Numbers / other scalars: stringify (e.g. a DOI emitted as a number).
    return str(value)


def _as_int(value: Any) -> int | None:
    """Coerce to int when unambiguous (int, or a clean integer string/float)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            # A year embedded in a longer string is not reliably extractable;
            # leave it null (EH-4) rather than guess.
            return None
    return None


def _normalize_authors(value: Any) -> list[str]:
    """Normalize the authors field into a source-ordered list of names.

    Tolerates a list of strings, a list of ``{"name": ...}`` dicts (and common
    ``given``/``family`` splits), or a single delimited string. Order is
    preserved (authors are a source-order collection, never sorted, NFR-1).
    """
    if value is None:
        return []
    if isinstance(value, str):
        # A single "A. Author, B. Scientist" style string.
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            name = _author_name(item)
            if name:
                out.append(name)
        return out
    return []


def _author_name(item: Any) -> str | None:
    """Extract a single author display name from a string or dict entry."""
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        # Direct name fields, then nested author object (OpenAlex-style
        # authorships), then given/family composition.
        for key in ("name", "display_name", "fullName", "full_name"):
            if isinstance(item.get(key), str) and item[key].strip():
                return item[key].strip()
        nested = item.get("author")
        if isinstance(nested, dict):
            name = _author_name(nested)
            if name:
                return name
        given = item.get("given") or item.get("givenName")
        family = item.get("family") or item.get("familyName") or item.get("last")
        composed = " ".join(p.strip() for p in (given, family) if isinstance(p, str) and p.strip())
        return composed or None
    return None


def _is_code_host_url(url: str) -> bool:
    """True if ``url`` points at a known code-hosting domain (S8.1)."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        # Malformed URL: treat as not a code link (EH-10; never raise here).
        return False
    if not host:
        return False
    return any(host == suffix or host.endswith("." + suffix) for suffix in _CODE_HOST_SUFFIXES)


def _extract_code_url(raw: dict[str, Any]) -> str | None:
    """Find an explicit metadata code link (S8.1), checking, in order:

    1. dedicated code/repository fields,
    2. a Papers-With-Code style ``externalIds``/``external_ids`` map,
    3. any generic URL field whose host is a known code host.

    Returns the raw link string (or None). Stage 7 makes the final, deterministic
    selection when several links exist; this only surfaces a candidate link.
    """
    # 1. Dedicated code fields.
    code = _as_str(_first(raw, _CODE_KEYS))
    if code:
        return code
    # 2. externalIds / external_ids nested map (Semantic-Scholar-ish shapes).
    for ext_key in ("externalIds", "external_ids", "externalIDs"):
        ext = raw.get(ext_key)
        if isinstance(ext, dict):
            for cand in ext.values():
                cand_str = _as_str(cand)
                if cand_str and _is_code_host_url(cand_str):
                    return cand_str
    # 3. Generic URL fields that happen to be code hosts.
    for key in _URL_KEYS:
        url = _as_str(raw.get(key))
        if url and _is_code_host_url(url):
            return url
    return None


def _detect_preprint(
    raw: dict[str, Any], venue_raw: str | None, arxiv_id: str | None
) -> bool:
    """Best-effort preprint detection (used by the Stage 4 preprint exception).

    A candidate is a preprint if an explicit flag says so, if its venue names a
    known preprint server, or if it carries an arXiv id and no published venue.
    """
    for key in ("is_preprint", "preprint", "isPreprint"):
        flag = raw.get(key)
        if isinstance(flag, bool):
            if flag:
                return True
            # An explicit False is authoritative.
            return False
    venue_l = (venue_raw or "").lower()
    if any(token in venue_l for token in ("arxiv", "preprint", "biorxiv", "medrxiv", "ssrn")):
        return True
    if arxiv_id and not venue_raw:
        return True
    return False


def normalize_result(raw: dict[str, Any], source: str, query: str) -> PaperCandidate:
    """Map one raw source result dict into a :class:`PaperCandidate` (S5 Stage 3 entry).

    Tolerant per EH-4: any field missing from ``raw`` becomes ``None`` (or an empty
    list for authors). Detects ``is_preprint`` and captures ``metadata_code_url``
    from explicit code/repository fields or known code-host URLs (S8.1). ``source``
    is the logical label ("papersflow" | "paper_search"); ``query`` is the query
    that surfaced this result (recorded in ``found_by_queries``).

    The ``candidate_id`` is a deterministic, source-stable identifier derived from
    the strongest available signal (DOI > arXiv id > normalized title) so the same
    raw result always yields the same id regardless of input ordering (NFR-1).
    """
    title = _as_str(_first(raw, _TITLE_KEYS)) or ""
    abstract = _as_str(_first(raw, _ABSTRACT_KEYS))
    venue_raw = _as_str(_first(raw, _VENUE_KEYS))
    year = _as_int(_first(raw, _YEAR_KEYS))
    citation_count = _as_int(_first(raw, _CITATION_KEYS))
    doi = _as_str(_first(raw, _DOI_KEYS))
    arxiv_id = _as_str(_first(raw, _ARXIV_KEYS))
    paper_url = _as_str(_first(raw, _URL_KEYS))
    language = _as_str(_first(raw, _LANG_KEYS))
    authors = _normalize_authors(_first(raw, _AUTHOR_KEYS))
    code_url = _extract_code_url(raw)
    is_preprint = _detect_preprint(raw, venue_raw, arxiv_id)

    candidate_id = _make_candidate_id(source, doi, arxiv_id, title)

    return PaperCandidate(
        candidate_id=candidate_id,
        title=title,
        authors=authors,
        abstract=abstract,
        venue_raw=venue_raw,
        venue_canonical=None,
        year=year,
        citation_count=citation_count,
        doi=doi,
        arxiv_id=arxiv_id,
        paper_url=paper_url,
        metadata_code_url=code_url,
        language=language,
        is_preprint=is_preprint,
        source=source,  # type: ignore[arg-type]  # router only passes valid labels
        found_by_queries=[query],
    )


def _normalize_title_key(title: str) -> str:
    """Lowercase + whitespace-collapsed title for stable id derivation."""
    return " ".join(title.lower().split())


def _make_candidate_id(
    source: str, doi: str | None, arxiv_id: str | None, title: str
) -> str:
    """Deterministic candidate id from the strongest available signal.

    Prefers DOI, then arXiv id, then a normalized title; prefixed by source so two
    sources' records stay distinct before dedup. The id is internal (pre-rank) and
    never emitted, so a readable derived form is sufficient and fully reproducible.
    """
    if doi:
        body = f"doi:{doi.lower()}"
    elif arxiv_id:
        body = f"arxiv:{arxiv_id.lower()}"
    else:
        body = f"title:{_normalize_title_key(title)}"
    return f"{source}:{body}"


def _parse_results(raw_response: str) -> list[dict[str, Any]]:
    """Parse a tool's JSON response into a list of result dicts (tolerant).

    Accepts: a JSON array of objects; an object wrapping the array under a common
    key (``results``/``papers``/``data``/``items``/``hits``); or a single object
    (treated as a one-element list). Anything unparseable yields an empty list and
    is logged -- a malformed source response degrades coverage, it does not crash
    Stage 2 (EH-2 spirit).
    """
    try:
        obj = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("could not parse tool response as JSON; skipping: %s", exc)
        return []
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        for key in ("results", "papers", "data", "items", "hits"):
            inner = obj.get(key)
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
        # A bare single-result object.
        return [obj]
    return []


def _resolve_source_tool(router: ToolRouter, tool_names: tuple[str, ...]) -> str | None:
    """First advertised tool name for a source, or None if the source is absent."""
    for name in tool_names:
        if router.has(name):
            return name
    return None


def search_candidates(
    router: ToolRouter,
    queries: list[str],
    max_results_per_source: int,
) -> list[PaperCandidate]:
    """Run every query against every available source and return raw candidates (S2).

    For each query, each source's resolved search tool is called with the query and
    a result cap; up to ``max_results_per_source`` results per source per query are
    normalized and kept (over-cap results are truncated deterministically by their
    returned order). Each candidate is tagged with its ``source`` and the surfacing
    ``query`` (``found_by_queries``).

    Coverage degrades gracefully: a source whose tools are absent is skipped (EH-2);
    a per-call tool error is logged and skipped. If NO source provides a search tool
    at all, raises :class:`AnveshaError` (EH-3 -> fatal at the orchestrator). The
    returned list is the ``identified`` count basis (raw, pre-dedup).
    """
    # Resolve which sources are actually available up front (deterministic order).
    available: list[tuple[str, str]] = []
    for source_label, tool_names in _SOURCES:
        tool = _resolve_source_tool(router, tool_names)
        if tool is None:
            logger.warning(
                "search source %r unavailable (none of %s advertised); skipping",
                source_label,
                ", ".join(tool_names),
            )
            continue
        available.append((source_label, tool))

    if not available:
        # EH-3: no discovery source at all is fatal.
        raise AnveshaError(
            "no search tool available from any discovery source "
            f"(tried: {', '.join(n for _, ns in _SOURCES for n in ns)})"
        )

    cap = max(0, max_results_per_source)
    candidates: list[PaperCandidate] = []
    succeeded_sources: set[str] = set()
    for query in queries:
        for source_label, tool_name in available:
            arguments = {"query": query, "max_results": cap}
            try:
                raw_response = router.call(tool_name, arguments)
            except Exception as exc:  # noqa: BLE001 - per-call failure degrades coverage (EH-2)
                logger.warning(
                    "search tool %r failed for query %r; skipping: %s",
                    tool_name,
                    query,
                    exc,
                )
                continue
            succeeded_sources.add(source_label)
            results = _parse_results(raw_response)
            # ``cap`` is the literal per-source-per-query ceiling (0 -> none).
            if len(results) > cap:
                results = results[:cap]
            for raw in results:
                candidates.append(normalize_result(raw, source_label, query))

    # EH-3: if we attempted searches but EVERY available source errored on
    # EVERY call, that is total source failure (fatal), not a legitimate empty
    # result. A single source succeeding is EH-2 (degraded coverage), not EH-3.
    if queries and not succeeded_sources:
        raise AnveshaError(
            "all discovery sources failed at call time (EH-3): "
            f"{', '.join(label for label, _ in available)}"
        )
    return candidates
