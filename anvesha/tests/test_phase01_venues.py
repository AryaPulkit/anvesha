"""Tests for Phase 1 venue canonicalization (spec S7.4, Appendix B).

Offline: pure data/string logic, no network/MCP/LLM/GPU.
"""

from __future__ import annotations

import pytest

from anvesha.phases.phase01_filter.venues.alias_map import (
    DEFAULT_VENUE_ALIASES,
    VenueAlias,
    VenueAliasMap,
    normalize_venue_string,
)
from anvesha.phases.phase01_filter.venues.canonicalize import canonicalize_venue


@pytest.fixture()
def alias_map() -> VenueAliasMap:
    return VenueAliasMap()


# --- normalization -------------------------------------------------------


def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize_venue_string("  NeurIPS  \t Conference ") == "neurips conference"


def test_normalize_strips_punctuation_to_spaces():
    assert normalize_venue_string("IEEE/CVF") == "ieee cvf"


def test_normalize_all_punctuation_is_empty():
    assert normalize_venue_string("---") == ""


# --- exact canonical-key match ------------------------------------------


def test_exact_canonical_key_match(alias_map: VenueAliasMap):
    assert canonicalize_venue("NeurIPS", alias_map) == "NeurIPS"
    assert canonicalize_venue("CVPR", alias_map) == "CVPR"
    assert canonicalize_venue("ICML", alias_map) == "ICML"
    assert canonicalize_venue("ICLR", alias_map) == "ICLR"
    assert canonicalize_venue("ACL", alias_map) == "ACL"
    assert canonicalize_venue("EMNLP", alias_map) == "EMNLP"
    assert canonicalize_venue("AAAI", alias_map) == "AAAI"


def test_canonical_key_is_case_insensitive(alias_map: VenueAliasMap):
    assert canonicalize_venue("neurips", alias_map) == "NeurIPS"
    assert canonicalize_venue("cvpr", alias_map) == "CVPR"


# --- alias match ---------------------------------------------------------


def test_alias_nips_maps_to_neurips(alias_map: VenueAliasMap):
    assert canonicalize_venue("NIPS", alias_map) == "NeurIPS"
    assert canonicalize_venue("nips", alias_map) == "NeurIPS"


# --- full-name match -----------------------------------------------------


def test_full_name_match(alias_map: VenueAliasMap):
    assert (
        canonicalize_venue(
            "Advances in Neural Information Processing Systems", alias_map
        )
        == "NeurIPS"
    )
    assert (
        canonicalize_venue("International Conference on Machine Learning", alias_map)
        == "ICML"
    )


# --- messy / real-world venue strings -----------------------------------


def test_messy_cvpr_full_official_name(alias_map: VenueAliasMap):
    raw = "IEEE/CVF Conference on Computer Vision and Pattern Recognition"
    assert canonicalize_venue(raw, alias_map) == "CVPR"


def test_messy_strings_with_punctuation_and_spacing(alias_map: VenueAliasMap):
    assert canonicalize_venue("  neurips  ", alias_map) == "NeurIPS"
    assert (
        canonicalize_venue(
            "Conference on Empirical Methods in Natural Language Processing.",
            alias_map,
        )
        == "EMNLP"
    )
    assert (
        canonicalize_venue(
            "Annual Meeting of the Association for Computational Linguistics",
            alias_map,
        )
        == "ACL"
    )


# --- ISSN match ----------------------------------------------------------


def test_issn_match(alias_map: VenueAliasMap):
    # NeurIPS proceedings ISSN, with and without hyphen.
    assert canonicalize_venue("1049-5258", alias_map) == "NeurIPS"
    assert canonicalize_venue("10495258", alias_map) == "NeurIPS"


# --- DBLP-key match ------------------------------------------------------


def test_dblp_key_exact_prefix(alias_map: VenueAliasMap):
    assert canonicalize_venue("conf/cvpr", alias_map) == "CVPR"
    assert canonicalize_venue("conf/nips", alias_map) == "NeurIPS"


def test_dblp_key_with_year_segment(alias_map: VenueAliasMap):
    assert canonicalize_venue("conf/cvpr/2024", alias_map) == "CVPR"
    assert canonicalize_venue("conf/nips/SmithJ24", alias_map) == "NeurIPS"


def test_dblp_neurips_alias_key(alias_map: VenueAliasMap):
    assert canonicalize_venue("conf/neurips/2023", alias_map) == "NeurIPS"


# --- unknown / null ------------------------------------------------------


def test_unknown_venue_returns_none(alias_map: VenueAliasMap):
    assert canonicalize_venue("Journal of Irreproducible Results", alias_map) is None
    assert canonicalize_venue("SIGGRAPH", alias_map) is None


def test_empty_and_whitespace_return_none(alias_map: VenueAliasMap):
    assert canonicalize_venue("", alias_map) is None
    assert canonicalize_venue("   ", alias_map) is None
    assert canonicalize_venue("---", alias_map) is None


def test_none_returns_none(alias_map: VenueAliasMap):
    assert canonicalize_venue(None, alias_map) is None


# --- resolve() parity & determinism --------------------------------------


def test_resolve_matches_canonicalize_for_known(alias_map: VenueAliasMap):
    assert alias_map.resolve("NIPS") == "NeurIPS"
    assert alias_map.resolve(None) is None
    assert alias_map.resolve("totally unknown venue") is None


def test_default_map_is_deterministic_across_constructions():
    a = VenueAliasMap()
    b = VenueAliasMap()
    samples = [
        "NIPS",
        "neurips",
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "conf/cvpr/2024",
        "1049-5258",
        "unknown",
        None,
    ]
    assert [canonicalize_venue(s, a) for s in samples] == [
        canonicalize_venue(s, b) for s in samples
    ]


# --- custom alias map ----------------------------------------------------


def test_custom_alias_map():
    custom = VenueAliasMap(
        [
            VenueAlias(
                canonical_key="KDD",
                full_names=["Knowledge Discovery and Data Mining"],
                aliases=["KDD"],
                dblp_keys=["conf/kdd"],
                venue_type="conference",
            )
        ]
    )
    assert canonicalize_venue("KDD", custom) == "KDD"
    assert canonicalize_venue("conf/kdd/2022", custom) == "KDD"
    # NeurIPS is not in the custom map.
    assert canonicalize_venue("NeurIPS", custom) is None


def test_default_aliases_cover_required_venues():
    keys = {v.canonical_key for v in DEFAULT_VENUE_ALIASES}
    assert {"NeurIPS", "ICML", "CVPR", "ICLR", "ACL", "EMNLP", "AAAI"} <= keys
