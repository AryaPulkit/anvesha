"""Venue alias map for Phase 1 venue canonicalization (spec S7.4, Appendix B).

A ``VenueAliasMap`` resolves an inconsistent venue string (e.g. "CVPR", a full
conference name, or a DBLP key) to a canonical venue key. The resolution order
is defined in S7.4 and implemented in ``canonicalize.py``; this module supplies
the data model (``VenueAlias``), the built-in dataset (``DEFAULT_VENUE_ALIASES``),
and the lookup structure (``VenueAliasMap``).

This module is self-contained: it imports nothing from other phase modules and
performs no I/O (boundary rules, charter S12).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

VenueType = Literal["conference", "journal", "workshop"]


class VenueAlias(BaseModel):
    """One canonical venue and every string that should resolve to it (Appendix B)."""

    canonical_key: str
    full_names: list[str] = []
    aliases: list[str] = []
    dblp_keys: list[str] = []
    issns: list[str] = []
    venue_type: VenueType = "conference"


def normalize_venue_string(raw: str) -> str:
    """Normalize a venue string for matching (S7.1/S7.4 normalization).

    Lowercase, Unicode NFC, strip punctuation, collapse whitespace runs to a
    single space, and trim. Returns "" for an all-punctuation/empty input.
    """
    text = unicodedata.normalize("NFC", raw).lower()
    # Replace any character that is not a letter, digit, or whitespace with a
    # space; this turns "ieee/cvf" into "ieee cvf" and drops stray punctuation.
    text = re.sub(r"[^0-9a-z\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_dblp_key(raw: str) -> str:
    """Normalize a DBLP key for matching (case-fold, trim).

    DBLP keys are slash-delimited (e.g. "conf/nips/2024") and are NOT subjected
    to the punctuation-stripping normalization used for human-readable names,
    because the slashes are structurally meaningful.
    """
    return unicodedata.normalize("NFC", raw).strip().lower()


def _normalize_issn(raw: str) -> str:
    """Normalize an ISSN to uppercase digits + optional X check char, hyphen-free."""
    text = unicodedata.normalize("NFC", raw).strip().upper()
    return re.sub(r"[^0-9X]+", "", text)


# Built-in venue dataset (S7.4: at least the major ML/CV/NLP venues). Aliases
# and full names are kept conservative and verifiable; ISSNs/DBLP keys are only
# included where they are standard and stable.
DEFAULT_VENUE_ALIASES: list[VenueAlias] = [
    VenueAlias(
        canonical_key="NeurIPS",
        full_names=[
            "Advances in Neural Information Processing Systems",
            "Conference on Neural Information Processing Systems",
            "Neural Information Processing Systems",
        ],
        aliases=["NeurIPS", "NIPS"],
        dblp_keys=["conf/nips", "conf/neurips"],
        issns=["1049-5258"],
        venue_type="conference",
    ),
    VenueAlias(
        canonical_key="ICML",
        full_names=["International Conference on Machine Learning"],
        aliases=["ICML"],
        dblp_keys=["conf/icml"],
        issns=["2640-3498"],
        venue_type="conference",
    ),
    VenueAlias(
        canonical_key="CVPR",
        full_names=[
            "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
            "IEEE Conference on Computer Vision and Pattern Recognition",
            "Conference on Computer Vision and Pattern Recognition",
            "Computer Vision and Pattern Recognition",
        ],
        aliases=["CVPR"],
        dblp_keys=["conf/cvpr"],
        issns=["1063-6919"],
        venue_type="conference",
    ),
    VenueAlias(
        canonical_key="ICLR",
        full_names=["International Conference on Learning Representations"],
        aliases=["ICLR"],
        dblp_keys=["conf/iclr"],
        issns=[],
        venue_type="conference",
    ),
    VenueAlias(
        canonical_key="ACL",
        full_names=[
            "Annual Meeting of the Association for Computational Linguistics",
            "Meeting of the Association for Computational Linguistics",
            "Association for Computational Linguistics",
        ],
        aliases=["ACL"],
        dblp_keys=["conf/acl"],
        issns=[],
        venue_type="conference",
    ),
    VenueAlias(
        canonical_key="EMNLP",
        full_names=[
            "Conference on Empirical Methods in Natural Language Processing",
            "Empirical Methods in Natural Language Processing",
        ],
        aliases=["EMNLP"],
        dblp_keys=["conf/emnlp"],
        issns=[],
        venue_type="conference",
    ),
    VenueAlias(
        canonical_key="AAAI",
        full_names=[
            "AAAI Conference on Artificial Intelligence",
            "Conference on Artificial Intelligence",
        ],
        aliases=["AAAI"],
        dblp_keys=["conf/aaai"],
        issns=[],
        venue_type="conference",
    ),
]


class VenueAliasMap:
    """Deterministic lookup from any venue string to its canonical key.

    Construction builds four lookup tables (name/alias, ISSN, DBLP-key, and the
    canonical key itself), each keyed by its normalized form. ``resolve`` applies
    the S7.4 order: known names/aliases/canonical key -> ISSN -> DBLP key -> None.

    Determinism: if two aliases normalize to the same string the FIRST entry in
    construction order wins, and a conflict is logged. Built-in order is fixed by
    ``DEFAULT_VENUE_ALIASES``, so default construction is fully deterministic.
    """

    def __init__(self, aliases: list[VenueAlias] | None = None) -> None:
        self.aliases: list[VenueAlias] = list(
            aliases if aliases is not None else DEFAULT_VENUE_ALIASES
        )
        # normalized name/alias/canonical-key -> canonical_key
        self._name_index: dict[str, str] = {}
        # normalized ISSN -> canonical_key
        self._issn_index: dict[str, str] = {}
        # normalized DBLP key -> canonical_key
        self._dblp_index: dict[str, str] = {}
        for entry in self.aliases:
            key = entry.canonical_key
            names = [key, *entry.full_names, *entry.aliases]
            for name in names:
                norm = normalize_venue_string(name)
                self._register(self._name_index, norm, key, "name/alias", name)
            for issn in entry.issns:
                norm_issn = _normalize_issn(issn)
                self._register(self._issn_index, norm_issn, key, "ISSN", issn)
            for dblp in entry.dblp_keys:
                norm_dblp = _normalize_dblp_key(dblp)
                self._register(self._dblp_index, norm_dblp, key, "DBLP key", dblp)

    @staticmethod
    def _register(
        index: dict[str, str], norm: str, canonical_key: str, kind: str, original: str
    ) -> None:
        """Insert a normalized lookup entry, first-wins on conflict (deterministic)."""
        if not norm:
            return
        existing = index.get(norm)
        if existing is not None and existing != canonical_key:
            logger.warning(
                "venue %s '%s' (normalized '%s') maps to both '%s' and '%s'; "
                "keeping '%s'",
                kind,
                original,
                norm,
                existing,
                canonical_key,
                existing,
            )
            return
        index.setdefault(norm, canonical_key)

    def resolve(self, raw: str | None) -> str | None:
        """Resolve a raw venue string to a canonical key, or None (S7.4 order).

        Order: normalize -> known names/aliases/canonical key -> ISSN -> DBLP key
        -> None. An empty/whitespace-only/None input resolves to None.
        """
        if raw is None:
            return None
        # 1. Name / alias / canonical-key match on the name-normalized form.
        norm = normalize_venue_string(raw)
        if norm:
            hit = self._name_index.get(norm)
            if hit is not None:
                return hit
        # 2. ISSN match (a raw value that is itself an ISSN string).
        norm_issn = _normalize_issn(raw)
        if norm_issn:
            hit = self._issn_index.get(norm_issn)
            if hit is not None:
                return hit
        # 3. DBLP-key match (slash-structured key, normalized separately).
        norm_dblp = _normalize_dblp_key(raw)
        if norm_dblp:
            hit = self._dblp_index.get(norm_dblp)
            if hit is not None:
                return hit
        return None
