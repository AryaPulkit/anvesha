"""Tests for Phase 1 Stages 7-9: repo detection, PDF fetch, summary.

All offline: fake LLMClient (scripted/raising), fake MCPClient + ToolRouter
(scripted JSON returns), and a tmp_path workspace. No network, no real MCP
servers, no GPU.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Sequence

import pytest

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient, MCPToolDef
from anvesha.phases.phase01_filter.schemas.candidate import (
    PaperCandidate,
    RetainedPaper,
)
from anvesha.phases.phase01_filter.schemas.config import Options
from anvesha.phases.phase01_filter.stages import pdf_fetch, repo_detect, summarize
from anvesha.phases.phase01_filter.stages.pdf_fetch import DOWNLOAD_TOOL
from anvesha.phases.phase01_filter.tools import ToolRouter
from anvesha.workspace.project import ResearchProject, init_workspace


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class ScriptedLLM(LLMClient):
    """Returns the next scripted content per call; or raises if scripted to."""

    name = "scripted"

    def __init__(self, contents: Sequence[str | Exception]) -> None:
        self._contents = list(contents)
        self.calls: list[str] = []

    def run(
        self,
        prompt: str,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> LLMResponse:
        self.calls.append(prompt)
        item = self._contents.pop(0) if self._contents else ""
        if isinstance(item, Exception):
            raise item
        return LLMResponse(content=item)


class ScriptedMCP:
    """In-memory MCPClient exposing the PDF tool; returns scripted strings."""

    def __init__(
        self,
        returns: Sequence[str | Exception],
        tool_name: str = DOWNLOAD_TOOL,
    ) -> None:
        self._returns = list(returns)
        self._tool_name = tool_name
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[MCPToolDef]:
        return [MCPToolDef(self._tool_name, "download a pdf", {})]

    def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        item = self._returns.pop(0) if self._returns else ""
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture()
def project(tmp_path: Path) -> ResearchProject:
    return init_workspace(tmp_path / "proj", "Demo")


def make_paper(**overrides) -> RetainedPaper:
    base = dict(paper_id="paper_001", rank=1, title="A Study of Agents")
    base.update(overrides)
    return RetainedPaper(**base)


def make_candidate(**overrides) -> PaperCandidate:
    base = dict(candidate_id="c1", title="A Study of Agents", source="papersflow")
    base.update(overrides)
    return PaperCandidate(**base)


# --------------------------------------------------------------------------- #
# Stage 7 -- Repository Detection (S8)
# --------------------------------------------------------------------------- #
def test_repo_explicit_metadata_link() -> None:
    paper = make_paper()
    cand = make_candidate(metadata_code_url="https://github.com/org/repo")
    repo_detect.detect_repository(paper, cand)
    assert paper.code_url == "https://github.com/org/repo"
    assert paper.git_exists is True


def test_repo_explicit_link_preferred_over_paper_url() -> None:
    paper = make_paper()
    cand = make_candidate(
        metadata_code_url="https://gitlab.com/org/a",
        paper_url="https://github.com/org/b",
    )
    repo_detect.detect_repository(paper, cand)
    assert paper.code_url == "https://gitlab.com/org/a"
    assert paper.git_exists is True


def test_repo_paper_url_known_host_detected() -> None:
    paper = make_paper()
    cand = make_candidate(paper_url="https://huggingface.co/org/model")
    repo_detect.detect_repository(paper, cand)
    assert paper.code_url == "https://huggingface.co/org/model"
    assert paper.git_exists is True


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/x/y",
        "https://www.github.com/x/y",
        "https://gitlab.com/x/y",
        "https://bitbucket.org/x/y",
        "https://huggingface.co/datasets/x",
        "https://codeberg.org/x/y",
    ],
)
def test_repo_all_known_hosts(url: str) -> None:
    paper = make_paper()
    repo_detect.detect_repository(paper, make_candidate(paper_url=url))
    assert paper.git_exists is True
    assert paper.code_url == url


def test_repo_non_code_paper_url_ignored() -> None:
    paper = make_paper()
    cand = make_candidate(paper_url="https://arxiv.org/abs/2401.00001")
    repo_detect.detect_repository(paper, cand)
    assert paper.code_url is None
    assert paper.git_exists is False


def test_repo_no_metadata_sets_false() -> None:
    paper = make_paper()
    repo_detect.detect_repository(paper, make_candidate())
    assert paper.code_url is None
    assert paper.git_exists is False


@pytest.mark.parametrize(
    "bad",
    [
        "not a url",
        "://missing-scheme",
        "github.com/org/repo",  # no scheme -> not well-formed
        "ht!tp://[bad",
        "",
    ],
)
def test_repo_malformed_explicit_link_does_not_raise(bad: str) -> None:
    paper = make_paper()
    cand = make_candidate(metadata_code_url=bad)
    # EH-10: never throws; malformed explicit link ignored.
    repo_detect.detect_repository(paper, cand)
    assert paper.git_exists is False
    assert paper.code_url is None


def test_repo_malformed_explicit_then_valid_paper_url() -> None:
    paper = make_paper()
    cand = make_candidate(
        metadata_code_url="not a url",
        paper_url="https://github.com/org/fallback",
    )
    repo_detect.detect_repository(paper, cand)
    assert paper.code_url == "https://github.com/org/fallback"
    assert paper.git_exists is True


def test_repo_invariant_always_holds() -> None:
    for cand in (
        make_candidate(metadata_code_url="https://github.com/a/b"),
        make_candidate(paper_url="https://example.com/paper"),
        make_candidate(),
    ):
        paper = make_paper()
        repo_detect.detect_repository(paper, cand)
        assert paper.git_exists == (paper.code_url is not None)


# --------------------------------------------------------------------------- #
# Stage 8 -- PDF Acquisition (S9)
# --------------------------------------------------------------------------- #
def _router(returns, tool_name: str = DOWNLOAD_TOOL) -> ToolRouter:
    return ToolRouter([ScriptedMCP(returns, tool_name=tool_name)])


def test_pdf_skipped_when_download_disabled(project: ResearchProject) -> None:
    paper = make_paper()
    mcp = ScriptedMCP(["whatever"])
    router = ToolRouter([mcp])
    opts = Options(pdf_download=False)
    pdf_fetch.fetch_pdf(router, paper, project, "out/pdfs/paper_001.pdf", opts)
    assert paper.pdf_status == "skipped"
    assert paper.pdf_path is None
    assert mcp.calls == []  # no tool call when disabled


def test_pdf_ok_from_base64_json(project: ResearchProject) -> None:
    pdf_bytes = b"%PDF-1.7\nhello"
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    router = _router(['{"content": "' + b64 + '"}'])
    paper = make_paper(doi="10.1/x", arxiv_id="2401.00001")
    opts = Options()
    pdf_fetch.fetch_pdf(router, paper, project, "out/pdfs/paper_001.pdf", opts)
    assert paper.pdf_status == "ok"
    assert paper.pdf_path == "pdfs/paper_001.pdf"
    written = project.resolve("out/pdfs/paper_001.pdf")
    assert written.is_file()
    assert written.read_bytes() == pdf_bytes


def test_pdf_ok_from_bare_base64(project: ResearchProject) -> None:
    pdf_bytes = b"%PDF-bare"
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    router = _router([b64])
    paper = make_paper()
    pdf_fetch.fetch_pdf(router, paper, project, "out/pdfs/paper_001.pdf", Options())
    assert paper.pdf_status == "ok"
    assert project.resolve("out/pdfs/paper_001.pdf").read_bytes() == pdf_bytes


def test_pdf_ok_from_path_return(project: ResearchProject, tmp_path: Path) -> None:
    src = tmp_path / "downloaded.pdf"
    src.write_bytes(b"%PDF-from-path")
    router = _router(['{"path": "' + str(src) + '"}'])
    paper = make_paper()
    pdf_fetch.fetch_pdf(router, paper, project, "out/pdfs/paper_001.pdf", Options())
    assert paper.pdf_status == "ok"
    assert paper.pdf_path == "pdfs/paper_001.pdf"
    assert project.resolve("out/pdfs/paper_001.pdf").read_bytes() == b"%PDF-from-path"


def test_pdf_failed_when_tool_returns_empty(project: ResearchProject) -> None:
    router = _router(["", "", ""])
    paper = make_paper()
    opts = Options(pdf_max_retries=3)
    pdf_fetch.fetch_pdf(router, paper, project, "out/pdfs/paper_001.pdf", opts)
    assert paper.pdf_status == "failed"
    assert paper.pdf_path is None
    assert not project.resolve("out/pdfs/paper_001.pdf").exists()


def test_pdf_retries_then_succeeds(project: ResearchProject) -> None:
    pdf_bytes = b"%PDF-retry"
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    mcp = ScriptedMCP(["", RuntimeError("transient"), b64])
    router = ToolRouter([mcp])
    paper = make_paper()
    opts = Options(pdf_max_retries=3)
    pdf_fetch.fetch_pdf(router, paper, project, "out/pdfs/paper_001.pdf", opts)
    assert paper.pdf_status == "ok"
    assert len(mcp.calls) == 3
    assert project.resolve("out/pdfs/paper_001.pdf").read_bytes() == pdf_bytes


def test_pdf_tool_raises_all_attempts_non_fatal(project: ResearchProject) -> None:
    mcp = ScriptedMCP([RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")])
    router = ToolRouter([mcp])
    paper = make_paper()
    opts = Options(pdf_max_retries=3)
    # EH-5: never raises.
    pdf_fetch.fetch_pdf(router, paper, project, "out/pdfs/paper_001.pdf", opts)
    assert paper.pdf_status == "failed"
    assert paper.pdf_path is None
    assert len(mcp.calls) == 3


def test_pdf_missing_tool_marks_failed(project: ResearchProject) -> None:
    # Router has a different tool only; DOWNLOAD_TOOL absent.
    router = _router(["x"], tool_name="paper-search.search_papers")
    paper = make_paper()
    pdf_fetch.fetch_pdf(router, paper, project, "out/pdfs/paper_001.pdf", Options())
    assert paper.pdf_status == "failed"
    assert paper.pdf_path is None


def test_pdf_arguments_priority_doi_arxiv_title(project: ResearchProject) -> None:
    mcp = ScriptedMCP([""])
    router = ToolRouter([mcp])
    paper = make_paper(doi="10.1/x", arxiv_id="2401.00001", title="My  Paper Title")
    pdf_fetch.fetch_pdf(router, paper, project, "out/pdfs/paper_001.pdf", Options())
    name, args = mcp.calls[0]
    assert name == DOWNLOAD_TOOL
    assert args["doi"] == "10.1/x"
    assert args["arxiv_id"] == "2401.00001"
    assert args["title"] == "my paper title"  # normalized (collapsed/lowered)


def test_pdf_invalid_base64_in_json_is_failure(project: ResearchProject) -> None:
    # JSON content present but not valid base64 -> no bytes -> failed.
    router = _router(['{"content": "@@@not-base64@@@"}'])
    paper = make_paper()
    pdf_fetch.fetch_pdf(router, paper, project, "out/pdfs/paper_001.pdf", Options())
    assert paper.pdf_status == "failed"


# --------------------------------------------------------------------------- #
# Stage 9 -- Summary Generation (S9 rubric)
# --------------------------------------------------------------------------- #
def test_summary_ok() -> None:
    body = "This paper addresses X. Methodology Y. Contributions Z. Build W. Useful."
    llm = ScriptedLLM([body])
    paper = make_paper()
    summarize.summarize_paper(llm, paper, "An abstract.", Options())
    assert paper.summary == body
    assert paper.summary_status == "ok"
    assert len(llm.calls) == 1
    # Prompt carries the title and abstract.
    assert "A Study of Agents" in llm.calls[0]
    assert "An abstract." in llm.calls[0]


def test_summary_strips_whitespace() -> None:
    llm = ScriptedLLM(["  padded summary  \n"])
    paper = make_paper()
    summarize.summarize_paper(llm, paper, "abs", Options())
    assert paper.summary == "padded summary"
    assert paper.summary_status == "ok"


def test_summary_retries_then_succeeds() -> None:
    llm = ScriptedLLM([RuntimeError("flaky"), "second try works"])
    paper = make_paper()
    summarize.summarize_paper(llm, paper, "abs", Options(summary_max_retries=2))
    assert paper.summary == "second try works"
    assert paper.summary_status == "ok"
    assert len(llm.calls) == 2


def test_summary_empty_response_is_failure() -> None:
    llm = ScriptedLLM(["", "   "])
    paper = make_paper()
    summarize.summarize_paper(llm, paper, "abs", Options(summary_max_retries=2))
    assert paper.summary == summarize.FAILED_SUMMARY
    assert paper.summary_status == "failed"


def test_summary_all_attempts_raise_is_failure() -> None:
    llm = ScriptedLLM([RuntimeError("a"), RuntimeError("b")])
    paper = make_paper()
    # EH-6: never raises.
    summarize.summarize_paper(llm, paper, "abs", Options(summary_max_retries=2))
    assert paper.summary == "Summary generation failed."
    assert paper.summary_status == "failed"
    assert len(llm.calls) == 2


def test_summary_none_abstract_handled() -> None:
    llm = ScriptedLLM(["ok summary"])
    paper = make_paper()
    summarize.summarize_paper(llm, paper, None, Options())
    assert paper.summary == "ok summary"
    assert "(no abstract available)" in llm.calls[0]
