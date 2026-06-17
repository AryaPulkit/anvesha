"""Anvesha command-line interface (ANVESHA.md S9).

Commands: init | run | resume | status | logs. Anvesha runs as a foreground
process; for long phases the researcher is expected to use tmux (S9). The CLI
never manages tmux itself.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from anvesha.core.config import PipelineConfig, load_config
from anvesha.core.exceptions import AnveshaError
from anvesha.workspace import ResearchProject, VersionLog, init_workspace

logger = logging.getLogger("anvesha")


def _find_project() -> ResearchProject:
    project = ResearchProject.find(".")
    project.validate()
    return project


def _build_phase_spec(config: PipelineConfig, phase_id: int):
    """Assemble a PhaseSpec from the resolved config for a registered phase."""
    from anvesha.core.phase import PhaseSpec

    entry = config.phase(phase_id)
    return PhaseSpec(
        phase_id=phase_id,
        name=PipelineConfig.phase_name(phase_id),
        inputs=entry.inputs,
        optional_inputs=entry.optional_inputs,
        output=entry.output,
        tools_permitted=entry.tools,
        options=entry.options,
        filters=entry.filters,
    )


def _run_phase(phase_id: int, backend_override: str | None, *, resume: bool) -> int:
    from anvesha.core.adapters import create_llm_client
    from anvesha.phases import PHASE_REGISTRY

    project = _find_project()
    name = PipelineConfig.phase_name(phase_id)

    phase_cls = PHASE_REGISTRY.get(phase_id)
    if phase_cls is None:
        # Unimplemented phases must not require backend config (per contract).
        print(f"Phase {phase_id} ({name}) is not implemented yet", file=sys.stderr)
        return 2

    config = load_config(project.root)
    llm = create_llm_client(config, phase_id, backend_override)
    spec = _build_phase_spec(config, phase_id)
    phase = phase_cls(project, VersionLog(project), llm, spec)
    verb = "Resuming" if resume else "Running"
    logger.info("%s phase %d (%s)", verb, phase_id, name)
    written = phase.run()
    print(f"Phase {phase_id} output written to {written}")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    project = init_workspace(args.path, args.name)
    print(f"Initialised Anvesha workspace {args.name!r} at {project.root}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    return _run_phase(args.phase, args.backend, resume=False)


def _cmd_resume(args: argparse.Namespace) -> int:
    return _run_phase(args.phase, None, resume=True)


def _cmd_status(args: argparse.Namespace) -> int:
    project = _find_project()
    version_log = VersionLog(project)
    if any(version_log.current(pid) is not None for pid in range(1, 12)):
        print(version_log.read().rstrip())
    else:
        print("no phases recorded")
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    project = _find_project()
    log_path = project.logs_dir / f"phase_{args.phase:02d}.log"
    if not log_path.is_file():
        print(f"No log file for phase {args.phase} at {log_path}", file=sys.stderr)
        return 1
    if args.follow:
        _follow(log_path)
    else:
        print(log_path.read_text(encoding="utf-8").rstrip())
    return 0


def _follow(log_path: Path, poll_seconds: float = 0.5) -> None:  # pragma: no cover
    """Print the file then poll for appended lines until interrupted."""
    with log_path.open("r", encoding="utf-8") as handle:
        sys.stdout.write(handle.read())
        sys.stdout.flush()
        try:
            while True:
                line = handle.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(poll_seconds)
        except KeyboardInterrupt:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anvesha", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create a new research project workspace")
    p_init.add_argument("--name", required=True, help="project name")
    p_init.add_argument("--path", default=".", help="workspace root (default: .)")
    p_init.set_defaults(func=_cmd_init)

    p_run = sub.add_parser("run", help="Run phase N")
    p_run.add_argument("--phase", type=int, required=True, choices=range(1, 12))
    p_run.add_argument("--backend", default=None, help="backend name override")
    p_run.set_defaults(func=_cmd_run)

    p_resume = sub.add_parser("resume", help="Resume phase N from last checkpoint")
    p_resume.add_argument("--phase", type=int, required=True, choices=range(1, 12))
    p_resume.set_defaults(func=_cmd_resume)

    p_status = sub.add_parser("status", help="Print the project version log")
    p_status.set_defaults(func=_cmd_status)

    p_logs = sub.add_parser("logs", help="Print or tail a phase log")
    p_logs.add_argument("--phase", type=int, required=True, choices=range(1, 12))
    p_logs.add_argument("--follow", action="store_true", help="tail the log")
    p_logs.set_defaults(func=_cmd_logs)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AnveshaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
