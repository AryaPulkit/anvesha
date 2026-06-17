"""Venue canonicalization for Phase 1 (spec S7.4, Appendix B).

Re-exports the public venue API. Modules elsewhere should still import by full
module path per the project import convention.
"""

from anvesha.phases.phase01_filter.venues.alias_map import (
    DEFAULT_VENUE_ALIASES,
    VenueAlias,
    VenueAliasMap,
    normalize_venue_string,
)
from anvesha.phases.phase01_filter.venues.canonicalize import canonicalize_venue

__all__ = [
    "DEFAULT_VENUE_ALIASES",
    "VenueAlias",
    "VenueAliasMap",
    "canonicalize_venue",
    "normalize_venue_string",
]
