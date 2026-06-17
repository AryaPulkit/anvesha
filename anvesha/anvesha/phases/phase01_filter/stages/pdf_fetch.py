"""Stage 8 -- PDF Acquisition (MCP), spec S9.

For each retained paper, when ``options.pdf_download`` is true, attempt a
download via paper-search-mcp's ``download_with_fallback`` tool, supplying
identifiers in priority order DOI -> arXiv ID -> normalized title, retrying up
to ``options.pdf_max_retries`` times. On success the bytes are written to
``pdfs/{paper_id}.pdf`` through the workspace I/O module and the record is
marked ``pdf_status = "ok"``; on failure the record is marked
``pdf_status = "failed"`` with ``pdf_path = None``. When ``pdf_download`` is
false the record is ``pdf_status = "skipped"``.

A failed download is non-fatal (NFR-5 / EH-5): this stage never raises -- any
tool error, parse error, or write error is logged and recorded as a failure.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from pathlib import Path

from anvesha.phases.phase01_filter.schemas.candidate import RetainedPaper
from anvesha.phases.phase01_filter.schemas.config import Options
from anvesha.phases.phase01_filter.tools import ToolRouter
from anvesha.workspace import phase_io
from anvesha.workspace.project import ResearchProject

logger = logging.getLogger(__name__)

# paper-search-mcp PDF tool (S9.1). Namespaced per the MCP convention.
DOWNLOAD_TOOL = "paper-search.download_with_fallback"

# Keys the tool's JSON result may use to carry base64 PDF bytes / a path.
_CONTENT_KEYS = ("content", "data", "pdf_base64", "base64", "bytes")
_PATH_KEYS = ("path", "pdf_path", "file", "filepath", "filename", "local_path")


def _normalize_title(title: str) -> str:
    """Lowercase + whitespace-collapsed title (S7.1 normalization, fallback id)."""
    return " ".join(title.lower().split())


def _build_arguments(paper: RetainedPaper) -> dict:
    """Build the ``download_with_fallback`` arguments in S9.1 priority order.

    DOI -> arXiv ID -> normalized title. All identifiers present are supplied
    (the tool falls back across them); ``paper_id`` is passed so a path-based
    return can be associated, but acquisition never depends on it.
    """
    args: dict[str, object] = {"paper_id": paper.paper_id}
    if paper.doi:
        args["doi"] = paper.doi
    if paper.arxiv_id:
        args["arxiv_id"] = paper.arxiv_id
    args["title"] = _normalize_title(paper.title)
    return args


def _decode_base64(text: str) -> bytes | None:
    """Decode a base64 string to bytes, or ``None`` if it is not valid base64."""
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return None


def _extract_bytes(raw: str) -> bytes | None:
    """Obtain PDF bytes from a tolerant ``download_with_fallback`` return value.

    The tool's contract is loose; this handles the shapes the spec allows:

    * A JSON object with a base64 ``content`` (or alias) field -> decoded bytes.
    * A JSON object with a filesystem ``path`` (or alias) field -> the file's
      bytes (read directly: the tool wrote it outside the workspace and we copy
      it in via the workspace module).
    * A bare base64 string -> decoded bytes.
    * A bare filesystem path to an existing file -> the file's bytes.

    Returns ``None`` when no bytes can be obtained (treated as a failure, EH-5).
    """
    stripped = raw.strip()
    if not stripped:
        return None

    obj: object = None
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        obj = None

    if isinstance(obj, dict):
        for key in _CONTENT_KEYS:
            value = obj.get(key)
            if isinstance(value, str) and value:
                decoded = _decode_base64(value)
                if decoded is not None:
                    return decoded
        for key in _PATH_KEYS:
            value = obj.get(key)
            if isinstance(value, str) and value:
                data = _read_path(value)
                if data is not None:
                    return data
        return None

    # Bare string: either base64 content or a path-like string.
    decoded = _decode_base64(stripped)
    if decoded is not None:
        return decoded
    return _read_path(stripped)


def _read_path(value: str) -> bytes | None:
    """Read bytes from a filesystem path string, or ``None`` if not readable."""
    try:
        path = Path(value)
        if path.is_file():
            return path.read_bytes()
    except (OSError, ValueError) as exc:
        logger.warning("Could not read tool-returned PDF path %r: %s", value, exc)
    return None


def fetch_pdf(
    router: ToolRouter,
    paper: RetainedPaper,
    project: ResearchProject,
    pdf_relpath: str,
    options: Options,
) -> None:
    """Acquire and persist the PDF for one retained paper (S8 Stage8 / S9).

    Mutates ``paper`` in place, setting ``pdf_status`` and ``pdf_path``:

    * ``options.pdf_download is False`` -> ``pdf_status = "skipped"``,
      ``pdf_path = None`` (no tool call).
    * Otherwise attempt ``download_with_fallback`` (DOI -> arXiv -> title),
      retrying up to ``options.pdf_max_retries`` times. On the first attempt
      that yields bytes, write them to ``pdf_relpath`` via the workspace module
      and set ``pdf_path = "pdfs/{paper_id}.pdf"`` and ``pdf_status = "ok"``.
    * On exhausted retries, missing tool, or any error -> ``pdf_status =
      "failed"``, ``pdf_path = None`` (EH-5, non-fatal). ``pdf_relpath`` is the
      workspace-relative target (e.g. ``"<output_dir>/pdfs/paper_001.pdf"``);
      the emitted ``pdf_path`` is the run-dir-relative ``"pdfs/{paper_id}.pdf"``.

    Never raises.
    """
    if not options.pdf_download:
        paper.pdf_status = "skipped"
        paper.pdf_path = None
        return

    if not router.has(DOWNLOAD_TOOL):
        logger.warning(
            "PDF tool %s unavailable; marking %s failed (EH-5)",
            DOWNLOAD_TOOL,
            paper.paper_id,
        )
        paper.pdf_status = "failed"
        paper.pdf_path = None
        return

    arguments = _build_arguments(paper)
    # At least one attempt even if max_retries is <= 0; total attempts =
    # max(1, pdf_max_retries).
    attempts = max(1, options.pdf_max_retries)
    data: bytes | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = router.call(DOWNLOAD_TOOL, arguments)
        except Exception as exc:  # noqa: BLE001 - non-fatal per paper (EH-5)
            logger.warning(
                "PDF download attempt %d/%d for %s raised: %s",
                attempt,
                attempts,
                paper.paper_id,
                exc,
            )
            continue
        data = _extract_bytes(raw)
        if data:
            break
        logger.info(
            "PDF download attempt %d/%d for %s yielded no usable bytes",
            attempt,
            attempts,
            paper.paper_id,
        )

    if not data:
        paper.pdf_status = "failed"
        paper.pdf_path = None
        return

    try:
        phase_io.write_bytes_file(project, pdf_relpath, data)
    except Exception as exc:  # noqa: BLE001 - a write failure here is non-fatal
        # S9 download failures are non-fatal; a per-paper persist failure is
        # treated the same way (EH-5) rather than aborting the run. (A
        # whole-run disk failure surfaces as EH-9 at Stage 10 assembly.)
        logger.warning(
            "Failed to persist PDF for %s to %s: %s",
            paper.paper_id,
            pdf_relpath,
            exc,
        )
        paper.pdf_status = "failed"
        paper.pdf_path = None
        return

    paper.pdf_path = f"pdfs/{paper.paper_id}.pdf"
    paper.pdf_status = "ok"
