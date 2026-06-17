"""Stage 3 -- Normalize & Deduplicate (deterministic). Spec S7.

This module collapses duplicate :class:`PaperCandidate` records discovered
across sources. Duplicates are found by exact identifier match (DOI / arXiv id)
or by normalized-title similarity at or above ``options.dedup_threshold``
(S7.2). Grouping uses a similarity graph plus identifier equivalence and
connected components, so the surviving canonical record is identical
regardless of input order (S7.5). Each group is collapsed per the S7.3
conflict-resolution rules.

Pure / deterministic: no network, no LLM, no MCP, no clock, no randomness.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from anvesha.phases.phase01_filter.schemas.candidate import PaperCandidate, Source

logger = logging.getLogger(__name__)

# Hosts whose URLs count as explicit code links (S8.1 / S7.3 rule 3). Used here
# only to rank candidate code links deterministically when merging.
_CODE_HOSTS = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "huggingface.co",
    "codeberg.org",
)

# Matches any run of characters that is neither a Unicode word char nor space.
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


# --- normalization -------------------------------------------------------


def normalize_title(title: str | None) -> str:
    """Normalize a title for similarity comparison (S7.1).

    NFC-normalize, lowercase, replace punctuation with spaces, collapse
    internal whitespace to single spaces, and trim. Stopwords are NOT removed
    (S7.1: avoids collapsing distinct short titles). Returns ``""`` for a
    ``None``/empty title.
    """
    if not title:
        return ""
    text = unicodedata.normalize("NFC", title).lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _tokens(normalized: str) -> set[str]:
    """Token set of an already-normalized string (split on whitespace)."""
    return set(normalized.split())


def _jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two normalized titles, in [0, 1]."""
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union


def _levenshtein_distance(a: str, b: str) -> int:
    """Edit distance between two strings (no external deps).

    Standard two-row dynamic-programming implementation; O(len(a) * len(b))
    time and O(min) space.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Keep the inner loop over the shorter string for less memory.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,       # deletion
                current[j - 1] + 1,    # insertion
                previous[j - 1] + cost,  # substitution
            )
        previous = current
    return previous[-1]


def _levenshtein_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1] on normalized strings."""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - (_levenshtein_distance(a, b) / max_len)


def title_similarity(a: str, b: str) -> float:
    """Title similarity = max(token-set Jaccard, normalized Levenshtein) (S7.1).

    Inputs are raw titles; both are normalized internally. Result is a float
    in ``[0.0, 1.0]``.

    A missing title on either side yields ``0.0``: two papers with no title
    metadata share no title evidence and must not be merged on title
    similarity (exact DOI/arXiv matching still groups them when appropriate).
    Without this guard, two empty titles would score ``1.0`` and collapse
    distinct title-less papers, corrupting the dedup funnel (S7.1/S7.2, EH-4).
    """
    na = normalize_title(a)
    nb = normalize_title(b)
    if not na or not nb:
        return 0.0
    return max(_jaccard(na, nb), _levenshtein_ratio(na, nb))


# --- duplicate grouping --------------------------------------------------


def _norm_doi(doi: str | None) -> str | None:
    """Case-insensitive, whitespace-trimmed DOI key, or ``None`` if empty."""
    if not doi:
        return None
    key = doi.strip().lower()
    return key or None


def _norm_arxiv(arxiv_id: str | None) -> str | None:
    """Case-insensitive, whitespace-trimmed arXiv id key, or ``None`` if empty."""
    if not arxiv_id:
        return None
    key = arxiv_id.strip().lower()
    return key or None


def _build_components(
    candidates: list[PaperCandidate], dedup_threshold: float
) -> list[list[int]]:
    """Group candidate indices into duplicate sets via connected components.

    An edge connects i and j when their DOIs match (case-insensitive), their
    arXiv ids match, or their normalized-title similarity >= dedup_threshold
    (S7.2). Edges are undirected, so grouping is order-independent: the same
    components emerge for any input permutation. Components are returned with
    indices sorted ascending so downstream iteration is deterministic.
    """
    n = len(candidates)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            # Union toward the lower root for stable, order-independent roots.
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

    norm_titles = [normalize_title(c.title) for c in candidates]
    dois = [_norm_doi(c.doi) for c in candidates]
    arxivs = [_norm_arxiv(c.arxiv_id) for c in candidates]

    for i in range(n):
        for j in range(i + 1, n):
            if dois[i] is not None and dois[i] == dois[j]:
                union(i, j)
                continue
            if arxivs[i] is not None and arxivs[i] == arxivs[j]:
                union(i, j)
                continue
            # No title evidence on either side -> no title-similarity edge
            # (else two title-less papers would score 1.0 and wrongly merge).
            # Exact DOI/arXiv matching above still groups such papers correctly.
            if not norm_titles[i] or not norm_titles[j]:
                continue
            sim = max(
                _jaccard(norm_titles[i], norm_titles[j]),
                _levenshtein_ratio(norm_titles[i], norm_titles[j]),
            )
            if sim >= dedup_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(members) for members in groups.values()]


# --- conflict resolution (S7.3) ------------------------------------------


def _is_published(c: PaperCandidate) -> bool:
    """True if the member looks like a published (non-preprint) record (S7.3.1).

    A resolvable published venue is preferred over a preprint. ``venue_canonical``
    is set by the hard filter (later stage); during Stage 3 it is usually unset,
    so a non-preprint with any venue string also qualifies.
    """
    if c.is_preprint:
        return False
    return bool(c.venue_canonical) or bool(c.venue_raw)


def _code_link_rank(url: str) -> int:
    """Lower is better: explicit known code host (0) before generic URL (1)."""
    low = url.lower()
    return 0 if any(host in low for host in _CODE_HOSTS) else 1


def _choose_code_url(members: list[PaperCandidate]) -> str | None:
    """Pick the merged code link (S7.3.3): known-host links before generic URLs,
    ties broken by lexicographically smallest URL. ``None`` if none present."""
    links = [c.metadata_code_url.strip() for c in members if c.metadata_code_url and c.metadata_code_url.strip()]
    if not links:
        return None
    # Deterministic: sort by (host rank, url); first wins.
    links.sort(key=lambda u: (_code_link_rank(u), u))
    return links[0]


def _merge_group(members: list[PaperCandidate]) -> PaperCandidate:
    """Collapse a duplicate group into one canonical candidate (S7.3).

    ``members`` is already in stable (ascending original-index) order, which is
    used as the deterministic tie-break for otherwise-equal choices.
    """
    if len(members) == 1:
        return members[0]

    published = [c for c in members if _is_published(c)]
    # Venue/year donor (S7.3.1): a published member if any, else the first member.
    venue_donor = published[0] if published else members[0]

    # Identifiers (S7.3.2): retain any DOI and any arXiv id across members.
    doi = next((c.doi for c in members if c.doi), None)
    arxiv_id = next((c.arxiv_id for c in members if c.arxiv_id), None)

    # Code link (S7.3.3).
    code_url = _choose_code_url(members)

    # Abstract (S7.3.4): longest non-empty; ties keep the first such member.
    abstract: str | None = None
    best_abs_len = -1
    for c in members:
        if c.abstract:
            length = len(c.abstract)
            if length > best_abs_len:
                best_abs_len = length
                abstract = c.abstract

    # Authors (S7.3.5): longest author list; if equal length, the published
    # record's. Iterate members in stable order, prefer strictly longer; on a
    # tie adopt the donor's list only when the current pick is not already the
    # donor's. Simpler equivalent: pick max length, tie-break by published.
    authors = list(members[0].authors)
    best_len = len(authors)
    authors_from_published = _is_published(members[0])
    for c in members[1:]:
        clen = len(c.authors)
        c_pub = _is_published(c)
        if clen > best_len or (clen == best_len and c_pub and not authors_from_published):
            authors = list(c.authors)
            best_len = clen
            authors_from_published = c_pub

    # citation_count: keep the maximum known value (most complete metadata).
    citations = [c.citation_count for c in members if c.citation_count is not None]
    citation_count = max(citations) if citations else None

    # paper_url / language: prefer the venue donor's, then first non-null.
    paper_url = venue_donor.paper_url or next((c.paper_url for c in members if c.paper_url), None)
    language = venue_donor.language or next((c.language for c in members if c.language), None)

    # source (S7.3.6): "merged" when members differ; else the single source.
    distinct_sources = {c.source for c in members}
    source: Source = "merged" if len(distinct_sources) > 1 else next(iter(distinct_sources))

    # found_by_queries (S7.3.7): union across members. Internal/not emitted;
    # sorted lexicographically so the canonical record is byte-identical for
    # any input permutation (NFR-1 / S7.5 -- first-seen order would depend on
    # input order, breaking order-independence).
    found_by_queries = sorted({q for c in members for q in c.found_by_queries})

    # is_preprint: the merged record is a preprint only if every member is.
    is_preprint = all(c.is_preprint for c in members)

    # candidate_id: lexicographically smallest member id (deterministic, stable).
    candidate_id = min(c.candidate_id for c in members)

    return PaperCandidate(
        candidate_id=candidate_id,
        title=venue_donor.title,
        authors=authors,
        abstract=abstract,
        venue_raw=venue_donor.venue_raw,
        venue_canonical=venue_donor.venue_canonical,
        year=venue_donor.year,
        citation_count=citation_count,
        doi=doi,
        arxiv_id=arxiv_id,
        paper_url=paper_url,
        metadata_code_url=code_url,
        language=language,
        is_preprint=is_preprint,
        source=source,
        found_by_queries=found_by_queries,
    )


def deduplicate(
    candidates: list[PaperCandidate], dedup_threshold: float
) -> list[PaperCandidate]:
    """Deduplicate candidates into canonical records (S7.2, S7.3, S7.5).

    Groups candidates by exact identifier match or title similarity using
    connected components, then collapses each group per S7.3. The output is
    sorted by the canonical record's ``candidate_id`` so the surviving set is
    byte-identical for any input permutation (NFR-1 / S7.5).
    """
    if not candidates:
        return []

    components = _build_components(candidates, dedup_threshold)
    merged = [_merge_group([candidates[i] for i in members]) for members in components]
    merged.sort(key=lambda c: c.candidate_id)
    logger.info(
        "Deduplicated %d candidates into %d records (threshold=%.2f)",
        len(candidates),
        len(merged),
        dedup_threshold,
    )
    return merged
