"""Deterministic venue canonicalization (spec S7.4).

``canonicalize_venue`` resolves a raw venue string to a canonical key using the
S7.4 order delegated to ``VenueAliasMap.resolve``: normalize -> known
names/aliases/canonical key -> ISSN -> DBLP key -> None. It additionally peels a
trailing year/round segment off a DBLP key (e.g. "conf/cvpr/2024" -> "conf/cvpr")
so full DBLP record keys resolve via their venue prefix.

This module imports nothing from other phase modules and performs no I/O.
"""

from __future__ import annotations

import logging
import re

from anvesha.phases.phase01_filter.venues.alias_map import VenueAliasMap

logger = logging.getLogger(__name__)

# A DBLP record key looks like "conf/<venue>/<id>" or "journals/<venue>/<id>".
# The alias map stores the venue-level prefix ("conf/cvpr"); this matches a
# longer record key so we can strip the trailing segment(s) and retry.
_DBLP_KEY_RE = re.compile(r"^(conf|journals)/[^/]+(/.+)?$", re.IGNORECASE)


def canonicalize_venue(raw: str | None, alias_map: "VenueAliasMap") -> str | None:
    """Resolve ``raw`` to a canonical venue key, or None if unresolvable (S7.4).

    Resolution order (delegated to ``alias_map.resolve``):
      1. normalize and match against known names / aliases / canonical key
      2. match against ISSN
      3. match against DBLP key
      4. else None

    Extra DBLP handling: if a direct lookup fails and ``raw`` looks like a full
    DBLP record key (``conf/<venue>/<year>``), the trailing segment is dropped to
    try the venue-level prefix (``conf/<venue>``).
    """
    if raw is None:
        return None

    direct = alias_map.resolve(raw)
    if direct is not None:
        return direct

    stripped = _strip_dblp_suffix(raw)
    if stripped is not None and stripped != raw:
        prefix_hit = alias_map.resolve(stripped)
        if prefix_hit is not None:
            return prefix_hit

    return None


def _strip_dblp_suffix(raw: str) -> str | None:
    """Return the venue-level DBLP prefix for a full record key, else None.

    "conf/cvpr/2024" -> "conf/cvpr"; "conf/nips/SmithJ24" -> "conf/nips".
    Non-DBLP-shaped strings return None (no stripping attempted).
    """
    candidate = raw.strip()
    if "/" not in candidate or _DBLP_KEY_RE.match(candidate) is None:
        return None
    parts = candidate.split("/")
    if len(parts) < 3:
        return None
    # Keep the type segment and the venue segment only ("conf/<venue>").
    return "/".join(parts[:2])
