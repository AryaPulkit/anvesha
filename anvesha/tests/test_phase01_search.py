"""Tests for Phase 1 Stage 1 (Query Builder) and Stage 2 (Search).

Offline only -- no network, no MCP servers. A fake ``LLMClient`` returns scripted
structured output for the query builder; fake ``MCPClient`` instances feed a real
:class:`ToolRouter` scripted JSON tool responses for the searcher.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import pytest

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient, MCPToolDef
from anvesha.core.exceptions import AnveshaError
from anvesha.phases.phase01_filter.schemas.candidate import PaperCandidate
from anvesha.phases.phase01_filter.stages.query_builder import build_queries
from anvesha.phases.phase01_filter.stages.search import (
    normalize_result,
    search_candidates,
)
from anvesha.phases.phase01_filter.tools import ToolRouter


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeLLM(LLMClient):
    """Scripted LLM: ``structured_run`` returns ``structured`` (or raises)."""

    def __init__(self, structured: Any = None, *, raise_exc: Exception | None = None):
        self._structured = structured
        self._raise = raise_exc
        self.calls: list[str] = []

    def run(
        self,
        prompt: str,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Any] = (),
    ) -> LLMResponse:  # pragma: no cover - query builder uses structured_run
        return LLMResponse(content="")

    def structured_run(
        self,
        prompt: str,
        json_schema: dict,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Any] = (),
    ) -> dict:
        self.calls.append(prompt)
        if self._raise is not None:
            raise self._raise
        return self._structured


class FakeMCP(MCPClient):
    """Scripted MCP client: advertises ``tool_defs`` and replays ``responses``.

    ``responses`` maps a tool name to a JSON string; a per-tool call counter lets
    tests assert how often each tool was hit.
    """

    def __init__(
        self,
        tool_names: Sequence[str],
        responses: dict[str, str] | None = None,
        *,
        raise_on_list: bool = False,
        raise_on_call: set[str] | None = None,
    ):
        self._tool_names = list(tool_names)
        self._responses = responses or {}
        self._raise_on_list = raise_on_list
        self._raise_on_call = raise_on_call or set()
        self.call_log: list[tuple[str, dict]] = []

    def list_tools(self) -> list[MCPToolDef]:
        if self._raise_on_list:
            raise RuntimeError("list_tools boom")
        return [
            MCPToolDef(name=n, description="", input_schema={}) for n in self._tool_names
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        self.call_log.append((name, arguments))
        if name in self._raise_on_call:
            raise RuntimeError(f"{name} boom")
        return self._responses.get(name, "[]")


# --------------------------------------------------------------------------- #
# query_builder.build_queries
# --------------------------------------------------------------------------- #
def test_build_queries_happy_path_distinct_and_clean() -> None:
    llm = FakeLLM(
        {
            "queries": [
                "retrieval augmented generation",
                "LLM agent tool use",
                "memory compression agents",
            ]
        }
    )
    out = build_queries(llm, ["LLM Agents", "RAG"], None, max_queries=6)
    assert out == [
        "retrieval augmented generation",
        "LLM agent tool use",
        "memory compression agents",
    ]
    assert len(llm.calls) == 1


def test_build_queries_strips_embedded_year() -> None:
    # S5 Stage 1: years must never appear in query text; they are stripped.
    llm = FakeLLM({"queries": ["llm agents 2024 benchmarks"]})
    out = build_queries(llm, ["LLM Agents"], None, max_queries=6)
    assert out == ["llm agents benchmarks"]
    assert "2024" not in out[0]


def test_build_queries_drops_out_of_range_term_counts() -> None:
    # 1 term (too short) and 7 terms (too long) are dropped; 2-6 kept.
    llm = FakeLLM(
        {
            "queries": [
                "agents",  # 1 term -> drop
                "retrieval augmented generation for grounded agents reasoning",  # 7 -> drop
                "tool augmented agents",  # 3 -> keep
            ]
        }
    )
    out = build_queries(llm, ["LLM Agents"], None, max_queries=6)
    assert out == ["tool augmented agents"]


def test_build_queries_dedups_case_insensitively_preserving_order() -> None:
    llm = FakeLLM(
        {
            "queries": [
                "LLM Agents",
                "llm agents",  # dup (casefold) -> drop
                "retrieval augmented generation",
            ]
        }
    )
    out = build_queries(llm, ["LLM Agents"], None, max_queries=6)
    assert out == ["LLM Agents", "retrieval augmented generation"]


def test_build_queries_caps_at_max_queries() -> None:
    llm = FakeLLM(
        {"queries": [f"topic phrase {i}" for i in range(10)]}
    )
    out = build_queries(llm, ["LLM Agents"], None, max_queries=3)
    assert len(out) == 3
    assert out == ["topic phrase 0", "topic phrase 1", "topic phrase 2"]


def test_build_queries_zero_max_returns_empty() -> None:
    llm = FakeLLM({"queries": ["anything here"]})
    assert build_queries(llm, ["LLM Agents"], None, max_queries=0) == []


def test_build_queries_fallback_on_llm_failure() -> None:
    llm = FakeLLM(raise_exc=RuntimeError("model down"))
    out = build_queries(llm, ["LLM Agents", "Retrieval Augmented Generation"], None, 6)
    # Falls back to domain-derived queries (2-6 terms), config order preserved.
    assert out == ["LLM Agents", "Retrieval Augmented Generation"]


def test_build_queries_fallback_on_empty_llm_output() -> None:
    llm = FakeLLM({"queries": []})
    out = build_queries(llm, ["LLM Agents"], None, 6)
    assert out == ["LLM Agents"]


def test_build_queries_accepts_bare_list_shape() -> None:
    # Tolerant parsing: a bare list instead of {"queries": [...]}.
    llm = FakeLLM(["tool augmented agents", "graph retrieval methods"])
    out = build_queries(llm, ["LLM Agents"], None, 6)
    assert out == ["tool augmented agents", "graph retrieval methods"]


def test_build_queries_deterministic_repeat() -> None:
    payload = {"queries": ["retrieval augmented generation", "llm agent planning"]}
    a = build_queries(FakeLLM(dict(payload)), ["LLM Agents"], None, 6)
    b = build_queries(FakeLLM(dict(payload)), ["LLM Agents"], None, 6)
    assert a == b


def test_build_queries_passes_topic_statement_into_prompt() -> None:
    llm = FakeLLM({"queries": ["llm agent planning"]})
    build_queries(llm, ["LLM Agents"], "How do agents plan multi-step tool use?", 6)
    assert "multi-step tool use" in llm.calls[0]


def test_build_queries_single_word_domain_fallback() -> None:
    # All domains are single words and the model fails: spare single-term fallback
    # still produces a runnable query so Stage 2 is never starved.
    llm = FakeLLM(raise_exc=RuntimeError("down"))
    out = build_queries(llm, ["RAG"], None, 6)
    assert out == ["RAG"]


# --------------------------------------------------------------------------- #
# search.normalize_result
# --------------------------------------------------------------------------- #
def test_normalize_result_full_record() -> None:
    raw = {
        "title": "Toolformer Self-Supervised Tool Use",
        "authors": ["A. Researcher", {"name": "B. Scientist"}],
        "abstract": "We propose a self-supervised objective.",
        "venue": "NeurIPS",
        "year": 2024,
        "citation_count": 42,
        "doi": "10.0000/neurips.2024.00001",
        "arxiv_id": "2401.00001",
        "url": "https://example.org/abs/2024.00001",
        "code": "https://github.com/example/toolformer",
        "language": "en",
    }
    cand = normalize_result(raw, "papersflow", "tool use agents")
    assert isinstance(cand, PaperCandidate)
    assert cand.title == "Toolformer Self-Supervised Tool Use"
    assert cand.authors == ["A. Researcher", "B. Scientist"]
    assert cand.abstract == "We propose a self-supervised objective."
    assert cand.venue_raw == "NeurIPS"
    assert cand.year == 2024
    assert cand.citation_count == 42
    assert cand.doi == "10.0000/neurips.2024.00001"
    assert cand.arxiv_id == "2401.00001"
    assert cand.paper_url == "https://example.org/abs/2024.00001"
    assert cand.metadata_code_url == "https://github.com/example/toolformer"
    assert cand.language == "en"
    assert cand.source == "papersflow"
    assert cand.found_by_queries == ["tool use agents"]
    assert cand.is_preprint is False


def test_normalize_result_missing_fields_become_none() -> None:
    # EH-4: a near-empty record still yields a valid candidate with None fields.
    cand = normalize_result({"title": "Bare Title"}, "paper_search", "q")
    assert cand.title == "Bare Title"
    assert cand.authors == []
    assert cand.abstract is None
    assert cand.venue_raw is None
    assert cand.year is None
    assert cand.citation_count is None
    assert cand.doi is None
    assert cand.arxiv_id is None
    assert cand.paper_url is None
    assert cand.metadata_code_url is None
    assert cand.language is None
    assert cand.source == "paper_search"


def test_normalize_result_detects_arxiv_preprint() -> None:
    cand = normalize_result(
        {"title": "A Preprint", "arxiv_id": "2505.00042"}, "paper_search", "q"
    )
    assert cand.is_preprint is True


def test_normalize_result_detects_preprint_from_venue() -> None:
    cand = normalize_result(
        {"title": "X", "venue": "arXiv preprint arXiv:2401.1"}, "paper_search", "q"
    )
    assert cand.is_preprint is True


def test_normalize_result_explicit_preprint_false_wins() -> None:
    cand = normalize_result(
        {"title": "X", "arxiv_id": "2401.1", "is_preprint": False, "venue": "NeurIPS"},
        "papersflow",
        "q",
    )
    assert cand.is_preprint is False


def test_normalize_result_code_url_from_external_ids() -> None:
    raw = {
        "title": "X",
        "externalIds": {"DBLP": "conf/x", "PWC": "https://github.com/org/repo"},
    }
    cand = normalize_result(raw, "papersflow", "q")
    assert cand.metadata_code_url == "https://github.com/org/repo"


def test_normalize_result_code_url_from_generic_url_host() -> None:
    # A generic url field pointing at a code host counts as a code link (S8.1).
    raw = {"title": "X", "url": "https://gitlab.com/group/proj"}
    cand = normalize_result(raw, "papersflow", "q")
    assert cand.metadata_code_url == "https://gitlab.com/group/proj"


def test_normalize_result_non_code_url_not_treated_as_code() -> None:
    raw = {"title": "X", "url": "https://example.org/paper"}
    cand = normalize_result(raw, "papersflow", "q")
    assert cand.metadata_code_url is None
    assert cand.paper_url == "https://example.org/paper"


def test_normalize_result_code_field_preferred_over_url() -> None:
    raw = {
        "title": "X",
        "code": "https://github.com/org/from-code-field",
        "url": "https://gitlab.com/org/from-url",
    }
    cand = normalize_result(raw, "papersflow", "q")
    assert cand.metadata_code_url == "https://github.com/org/from-code-field"


def test_normalize_result_authors_from_given_family() -> None:
    raw = {
        "title": "X",
        "authors": [{"given": "Ada", "family": "Lovelace"}, {"name": "Alan Turing"}],
    }
    cand = normalize_result(raw, "papersflow", "q")
    assert cand.authors == ["Ada Lovelace", "Alan Turing"]


def test_normalize_result_authors_from_delimited_string() -> None:
    cand = normalize_result(
        {"title": "X", "authors": "A. One, B. Two , C. Three"}, "papersflow", "q"
    )
    assert cand.authors == ["A. One", "B. Two", "C. Three"]


def test_normalize_result_year_from_string_and_float() -> None:
    assert normalize_result({"title": "X", "year": "2023"}, "papersflow", "q").year == 2023
    assert normalize_result({"title": "X", "year": 2023.0}, "papersflow", "q").year == 2023
    # Unparseable year -> None (EH-4), never a guess.
    assert normalize_result({"title": "X", "year": "spring 2023"}, "papersflow", "q").year is None


def test_normalize_result_alias_keys() -> None:
    raw = {
        "name": "Aliased Title",
        "summary": "abs via summary",
        "journal": "ICML",
        "publication_year": 2025,
        "citationCount": 7,
        "DOI": "10.1/x",
        "arxivId": "2501.1",
    }
    cand = normalize_result(raw, "paper_search", "q")
    assert cand.title == "Aliased Title"
    assert cand.abstract == "abs via summary"
    assert cand.venue_raw == "ICML"
    assert cand.year == 2025
    assert cand.citation_count == 7
    assert cand.doi == "10.1/x"
    assert cand.arxiv_id == "2501.1"


def test_normalize_result_malformed_code_url_ignored() -> None:
    # EH-10: a malformed URL must not raise; it just is not treated as a code link.
    raw = {"title": "X", "url": "http://[not-a-valid-host"}
    cand = normalize_result(raw, "papersflow", "q")
    assert cand.metadata_code_url is None


def test_normalize_result_candidate_id_deterministic() -> None:
    raw = {"title": "X", "doi": "10.1/ABC"}
    a = normalize_result(raw, "papersflow", "q1")
    b = normalize_result(raw, "papersflow", "q2")
    assert a.candidate_id == b.candidate_id  # id is query-independent
    assert "10.1/abc" in a.candidate_id  # DOI lowercased into the id


# --------------------------------------------------------------------------- #
# search.search_candidates
# --------------------------------------------------------------------------- #
def _paper(title: str, **extra: Any) -> dict[str, Any]:
    return {"title": title, **extra}


def test_search_candidates_both_sources_tagged() -> None:
    pf = FakeMCP(
        ["papersflow.search"],
        {"papersflow.search": json.dumps([_paper("PF One"), _paper("PF Two")])},
    )
    ps = FakeMCP(
        ["paper-search.search_papers"],
        {"paper-search.search_papers": json.dumps([_paper("PS One")])},
    )
    router = ToolRouter([pf, ps])
    out = search_candidates(router, ["q1"], max_results_per_source=100)
    assert [c.title for c in out] == ["PF One", "PF Two", "PS One"]
    assert [c.source for c in out] == ["papersflow", "papersflow", "paper_search"]
    assert all(c.found_by_queries == ["q1"] for c in out)


def test_search_candidates_cap_per_source_per_query() -> None:
    many = json.dumps([_paper(f"P{i}") for i in range(10)])
    pf = FakeMCP(["papersflow.search"], {"papersflow.search": many})
    router = ToolRouter([pf])
    out = search_candidates(router, ["q1"], max_results_per_source=3)
    assert len(out) == 3
    assert [c.title for c in out] == ["P0", "P1", "P2"]
    # The cap is forwarded to the tool as max_results too.
    assert pf.call_log[0][1]["max_results"] == 3


def test_search_candidates_runs_each_query_against_each_source() -> None:
    pf = FakeMCP(["papersflow.search"], {"papersflow.search": json.dumps([_paper("P")])})
    router = ToolRouter([pf])
    search_candidates(router, ["q1", "q2", "q3"], max_results_per_source=10)
    queries = [args["query"] for _, args in pf.call_log]
    assert queries == ["q1", "q2", "q3"]


def test_search_candidates_query_order_then_source_order() -> None:
    pf = FakeMCP(["papersflow.search"], {"papersflow.search": json.dumps([_paper("pf")])})
    ps = FakeMCP(
        ["paper-search.search_papers"],
        {"paper-search.search_papers": json.dumps([_paper("ps")])},
    )
    router = ToolRouter([ps, pf])  # inject paper-search first; order must NOT depend on this
    out = search_candidates(router, ["q1", "q2"], max_results_per_source=10)
    # Per query: papersflow before paper_search (fixed source order in the stage).
    assert [(c.found_by_queries[0], c.source) for c in out] == [
        ("q1", "papersflow"),
        ("q1", "paper_search"),
        ("q2", "papersflow"),
        ("q2", "paper_search"),
    ]


def test_search_candidates_one_source_absent_degrades() -> None:
    # Only paper-search is present; papersflow is skipped (EH-2), not fatal.
    ps = FakeMCP(
        ["paper-search.search_arxiv"],
        {"paper-search.search_arxiv": json.dumps([_paper("PS Only")])},
    )
    router = ToolRouter([ps])
    out = search_candidates(router, ["q1"], max_results_per_source=10)
    assert [c.title for c in out] == ["PS Only"]
    assert out[0].source == "paper_search"


def test_search_candidates_no_source_raises() -> None:
    # EH-3: no discovery tool at all is fatal.
    other = FakeMCP(["filesystem.read"], {})
    router = ToolRouter([other])
    with pytest.raises(AnveshaError):
        search_candidates(router, ["q1"], max_results_per_source=10)


def test_search_candidates_per_call_error_skipped() -> None:
    # papersflow raises on call; paper-search still contributes (EH-2).
    pf = FakeMCP(["papersflow.search"], {}, raise_on_call={"papersflow.search"})
    ps = FakeMCP(
        ["paper-search.search_papers"],
        {"paper-search.search_papers": json.dumps([_paper("PS One")])},
    )
    router = ToolRouter([pf, ps])
    out = search_candidates(router, ["q1"], max_results_per_source=10)
    assert [c.title for c in out] == ["PS One"]


def test_search_candidates_parses_wrapped_results_key() -> None:
    pf = FakeMCP(
        ["papersflow.search"],
        {"papersflow.search": json.dumps({"results": [_paper("Wrapped")]})},
    )
    router = ToolRouter([pf])
    out = search_candidates(router, ["q1"], max_results_per_source=10)
    assert [c.title for c in out] == ["Wrapped"]


def test_search_candidates_parses_single_object_response() -> None:
    pf = FakeMCP(
        ["papersflow.search"],
        {"papersflow.search": json.dumps(_paper("Single"))},
    )
    router = ToolRouter([pf])
    out = search_candidates(router, ["q1"], max_results_per_source=10)
    assert [c.title for c in out] == ["Single"]


def test_search_candidates_malformed_json_skipped() -> None:
    pf = FakeMCP(["papersflow.search"], {"papersflow.search": "not json at all"})
    router = ToolRouter([pf])
    out = search_candidates(router, ["q1"], max_results_per_source=10)
    assert out == []


def test_search_candidates_prefers_first_advertised_tool_per_source() -> None:
    # papersflow advertises both tools; the stage uses the first in its order.
    pf = FakeMCP(
        ["papersflow.search", "papersflow.search_literature"],
        {
            "papersflow.search": json.dumps([_paper("from search")]),
            "papersflow.search_literature": json.dumps([_paper("from search_literature")]),
        },
    )
    router = ToolRouter([pf])
    out = search_candidates(router, ["q1"], max_results_per_source=10)
    assert [c.title for c in out] == ["from search"]


def test_search_candidates_deterministic_repeat() -> None:
    def make_router() -> ToolRouter:
        pf = FakeMCP(
            ["papersflow.search"],
            {"papersflow.search": json.dumps([_paper("A"), _paper("B")])},
        )
        ps = FakeMCP(
            ["paper-search.search_papers"],
            {"paper-search.search_papers": json.dumps([_paper("C")])},
        )
        return ToolRouter([pf, ps])

    a = [(c.title, c.source) for c in search_candidates(make_router(), ["q1", "q2"], 10)]
    b = [(c.title, c.source) for c in search_candidates(make_router(), ["q1", "q2"], 10)]
    assert a == b


def test_search_candidates_zero_cap_yields_nothing() -> None:
    pf = FakeMCP(
        ["papersflow.search"],
        {"papersflow.search": json.dumps([_paper("A"), _paper("B")])},
    )
    router = ToolRouter([pf])
    out = search_candidates(router, ["q1"], max_results_per_source=0)
    assert out == []
    assert pf.call_log[0][1]["max_results"] == 0
