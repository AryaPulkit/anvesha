"""Phase 1 entry point -- ``FilterLiteraturePipeline`` (spec S5, S4.1; charter S7).

This is the :class:`Phase` subclass the CLI/registry instantiates for Phase 1.
It owns the stages the orchestrator does not:

  * Stage 0 -- build and validate the resolved :class:`FilterConfig` /
    :class:`Options` from ``spec.filters`` / ``spec.options`` (EH-1 fatal on bad
    config), and read the optional topic statement from the first present input
    (``README.md``).
  * Output-directory naming + versioning (S4.1 / EH-8): construct the canonical
    ``filtered_literature[_v{N}][_{domain_slug}][_{years}]`` name and bump
    ``N`` until the directory is unused (unless ``overwrite``).
  * Drive :class:`FilterOrchestrator` (Stages 1-10).
  * ``validate_output`` the assembled document (S4.5) and, when valid, write
    ``01_filtered_literature.md`` into the run directory via the workspace
    module (PDFs were already placed under ``<dir>/pdfs/`` by Stage 8).
  * Stage 11 -- record the index path (workspace-relative) in the version log.

Determinism (NFR-1): ``generated_at`` is derived ONCE at run start
(``datetime.now(timezone.utc).isoformat()``) and threaded through the whole run
so the manifest is internally consistent. Determinism across runs is about
identical inputs + tool responses, not wall-clock equality.

Boundary rules (charter S12): all workspace I/O goes through ``phase_io`` /
``project.resolve``; the LLM and MCP clients are injected (never constructed);
the entry point receives a :class:`ResearchProject`. Phase 1 imports nothing
from other phase packages.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from anvesha.core.adapters.base import MCPClient
from anvesha.core.exceptions import PhaseValidationError
from anvesha.core.phase import Phase, PhaseSpec, ValidationResult
from anvesha.workspace import phase_io
from anvesha.workspace.project import ResearchProject
from anvesha.workspace.version_log import VersionLog

from anvesha.phases.phase01_filter.orchestrator import FilterOrchestrator
from anvesha.phases.phase01_filter.schemas.config import (
    FilterConfig,
    Options,
    build_filter_config,
    build_options,
)
from anvesha.phases.phase01_filter.state import PhaseState
from anvesha.phases.phase01_filter.stages import assemble
from anvesha.phases.phase01_filter.tools import ToolRouter

logger = logging.getLogger(__name__)

# Base segment of the output directory name (S4.1 rule 1).
_BASE_DIRNAME = "filtered_literature"
# The canonical index file written inside the run directory (S4.2).
_INDEX_FILENAME = "01_filtered_literature.md"
# Topic-statement marker in the workspace README skeleton; ignored when reading.
_TOPIC_PLACEHOLDER = "<!-- topic statement -->"


def _slugify(text: str) -> str:
    """Slugify a domain for the directory name (S4.1 rule 3).

    Lowercase; collapse non-alphanumeric runs to a single ``_``; strip leading
    and trailing ``_``. ``"LLM Agents"`` -> ``"llm_agents"``. Returns ``""`` for
    input with no alphanumeric content.
    """
    lowered = text.lower()
    collapsed = re.sub(r"[^a-z0-9]+", "_", lowered)
    return collapsed.strip("_")


class FilterLiteraturePipeline(Phase):
    """Phase 1 (Filter Literature) entry point (charter S7.1).

    ``__init__`` matches the :class:`Phase` base signature exactly so the CLI /
    registry can construct it uniformly. ``run`` is overridden because Phase 1
    is an 11-stage deterministic pipeline (not a single LLM call), and its output
    is a versioned *directory* (index + PDFs) rather than a single versioned file.
    """

    def __init__(
        self,
        project: ResearchProject,
        version_log: VersionLog,
        llm,
        spec: PhaseSpec,
    ) -> None:
        super().__init__(project, version_log, llm, spec)

    def task_description(self) -> str:
        """Brief task paragraph (charter S7.3). Phase 1 does not use the base prompt."""
        return (
            "Filter academic literature against the configured conferences, "
            "domains, and year range; deduplicate, score for relevance, rank, "
            "acquire PDFs, and emit a machine-consumable filtered-literature "
            "index (01_filtered_literature.md) with a PRISMA funnel."
        )

    def validate_output(self, content: str) -> ValidationResult:
        """Delegate to the Stage 10 validator (S4.5); wrap as a ValidationResult.

        ``assemble.validate_document`` returns a list of exit-criteria
        violations; an empty list means the document is valid.
        """
        errors = assemble.validate_document(content)
        return ValidationResult(ok=not errors, errors=errors)

    # -- Stage 0: resolved config + topic statement -----------------------
    def _build_config(self) -> tuple[FilterConfig, Options]:
        """Build and validate the FilterConfig/Options from the spec (Stage 0/EH-1).

        A bad filter block (e.g. ``years.start > years.end``, missing domains)
        raises a pydantic ``ValidationError`` here -- fatal, before any output
        directory is created (EH-1).
        """
        config = build_filter_config(self.spec.filters)
        options = build_options(self.spec.options)
        return config, options

    def _read_topic_statement(self) -> str | None:
        """Read the topic statement from the first present input (S3.3), if any.

        Returns the file's free-text topic (the whole file, trimmed), or ``None``
        when no input is present or the file carries no actual prose. The topic
        statement is an optional sharpening signal -- its absence is not an error.

        An untouched workspace README is the skeleton ``# {name}`` heading, a
        ``## Topic Statement`` heading, and the ``<!-- topic statement -->``
        placeholder. We treat that as *no* topic statement: after dropping the
        placeholder comment and any pure markdown heading / HTML-comment lines,
        if no prose remains the result is ``None``. As soon as the researcher
        adds real text, that text (the full trimmed file) is returned.
        """
        for relpath in self._present_inputs():
            if not phase_io.exists(self.project, relpath):
                continue
            text = phase_io.read_text(self.project, relpath).strip()
            if self._has_topic_prose(text):
                return text
        return None

    @staticmethod
    def _has_topic_prose(text: str) -> bool:
        """True if ``text`` contains topic prose beyond skeleton scaffolding.

        Lines that are pure markdown headings (``#`` ...), HTML comments, or the
        known placeholder are scaffolding; any other non-empty line is prose.
        """
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line == _TOPIC_PLACEHOLDER:
                continue
            if line.startswith("<!--") and line.endswith("-->"):
                continue
            return True
        return False

    # -- Output directory naming + versioning (S4.1 / EH-8) ---------------
    def _compose_dirname(self, config: FilterConfig, options: Options, version: int) -> str:
        """Compose the output-directory name for a given version (S4.1).

        ``filtered_literature[_v{N}][_{domain_slug}][_{year_start}_{year_end}]``.
        The version segment is always present (the base/options carry a version);
        domain and year segments are gated by the dirname options. ``output_root``
        is prepended as the parent directory (workspace-relative).
        """
        name = f"{_BASE_DIRNAME}_v{version}"
        if options.dirname_include_domain and config.domains:
            slug = _slugify(config.domains[0])
            if slug:
                name = f"{name}_{slug}"
        if options.dirname_include_years:
            name = f"{name}_{config.years.start}_{config.years.end}"
        root = options.output_root.strip() or "."
        if root in (".", "./", ""):
            return name
        # Normalize the parent into a workspace-relative posix path.
        return f"{Path(root).as_posix().rstrip('/')}/{name}"

    def _resolve_output_dirname(self, config: FilterConfig, options: Options) -> str:
        """Pick the versioned output-directory name, bumping N until unused (EH-8).

        Starts at ``options.output_version``. If the candidate directory already
        exists and ``overwrite`` is false, increment the version until an unused
        name is found (S4.1 rule 2). With ``overwrite`` true, the first candidate
        is used as-is. Probing is via ``phase_io.exists`` (workspace module only).
        """
        version = options.output_version
        name = self._compose_dirname(config, options, version)
        if options.overwrite:
            return name
        # Bounded probe loop; the bound is large enough to be effectively
        # unreachable in practice and guards against an infinite loop.
        for _ in range(10_000):
            if not phase_io.exists(self.project, name):
                return name
            version += 1
            name = self._compose_dirname(config, options, version)
        # Extremely unlikely fall-through: return the last computed name.
        return name

    # -- entry point ------------------------------------------------------
    def run(self, mcps: Sequence[MCPClient] = ()) -> Path:
        """Run Stages 0-11; return the absolute path of the written index file.

        Validates inputs, resolves config (Stage 0), names the versioned output
        directory (S4.1), runs the orchestrator (Stages 1-10), validates the
        assembled document (S4.5, raising :class:`PhaseValidationError` if it
        fails), writes ``01_filtered_literature.md`` into the directory, and
        records the workspace-relative index path in the version log (Stage 11).
        """
        # Validate required inputs exist (none are strictly required for Phase 1
        # since README is optional, but honor any declared required inputs).
        self.validate_inputs()

        # Stage 0: resolved configuration (EH-1 fatal on bad config).
        config, options = self._build_config()
        topic_statement = self._read_topic_statement()

        # Single timestamp for the whole run (internally consistent manifest).
        generated_at = datetime.now(timezone.utc).isoformat()

        # Output directory name with S4.1 versioning (EH-8).
        output_dir_name = self._resolve_output_dirname(config, options)
        logger.info(
            "Phase 1: output directory %s (overwrite=%s)",
            output_dir_name,
            options.overwrite,
        )

        # Build the run state and drive the orchestrator (Stages 1-10).
        state = PhaseState(
            config=config,
            options=options,
            topic_statement=topic_statement,
            output_dir_name=output_dir_name,
            generated_at=generated_at,
        )
        router = ToolRouter(mcps)
        orchestrator = FilterOrchestrator(self.llm, router, self.project, state)
        document = orchestrator.run()

        # Validate the assembled document against the S4.5 exit criteria.
        result = self.validate_output(document)
        if not result.ok:
            raise PhaseValidationError(
                f"Phase {self.spec.phase_id} ({self.spec.name}) output failed "
                f"validation: {'; '.join(result.errors)}"
            )

        # Stage 10 (persist): write the index inside the run directory (PDFs were
        # already placed under <dir>/pdfs/ by Stage 8). write_text_file writes the
        # EXACT path (no per-file versioning; the directory is the versioned unit).
        index_relpath = f"{output_dir_name}/{_INDEX_FILENAME}"
        written = phase_io.write_text_file(self.project, index_relpath, document)

        # Stage 11: record the active Phase 1 output in the version log.
        self.version_log.record(
            self.spec.phase_id,
            written.relative_to(self.project.root).as_posix(),
            note=f"retained={state.counts.retained}",
        )
        logger.info("Phase %d output written to %s", self.spec.phase_id, written)
        return written
