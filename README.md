# Anvesha — Research Assistant Kit for ML & CV Research

**Anvesha** (अन्वेषा, Sanskrit for *the seeking of knowledge*) is a phase-driven
research SDK that automates the mechanics of a deep-learning research workflow
end-to-end. It ships as a single Python package with a command-line interface;
each of eleven phases runs as a discrete command and produces a structured
Markdown file in the researcher's project workspace.

The authoritative design docs live in [`docs/`](docs/):

| Document | Scope |
|---|---|
| [`docs/ANVESHA.md`](docs/ANVESHA.md) | **Read this first** — project charter: architecture, the eleven phases, configuration, boundary rules |
| [`docs/filter_literature_implementation.md`](docs/filter_literature_implementation.md) | Phase 1 — Filter Literature |
| [`docs/literature_survey_implementation.md`](docs/literature_survey_implementation.md) | Phase 2 — Literature Survey |
| [`docs/research_gap_pipeline_implementation.md`](docs/research_gap_pipeline_implementation.md) | Phase 3 — Research Gaps |

## Status

| Component | State |
|---|---|
| Shared infrastructure (`core/`, `workspace/`) — LLM adapters, GPU scheduling, checkpointing, config, CLI, `Phase` base class | **Built**, tested |
| Phase 1 — Filter Literature | **Built**, tested |
| Phases 2–11 | Not implemented (specs exist for 2 and 3) |
| MCP transport layer (live PapersFlow / paper-search-mcp clients) | **Not yet built** — see caveat below |

Test suite: **403 tests passing.**

## Requirements

- Python **3.10+**
- For running phases against local LLMs: a GPU box serving Nemotron 3 Super /
  GPT-OSS-120B via vLLM (see `docs/ANVESHA.md` §8, §10). Not needed to install,
  run the test suite, or use `anvesha init` / `status`.

## Install

```bash
git clone <your-remote-url> anvesha_workspace
cd anvesha_workspace/anvesha

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # installs the package + dev (pytest) extras
```

The Python package lives in the `anvesha/` subdirectory (it holds
`pyproject.toml`); `docs/` and this README sit at the repository root.

## Verify the install

```bash
pytest -q                       # 403 tests should pass
anvesha --help
```

## Usage (CLI)

```bash
# create a research-project workspace
anvesha init --name "my-research-project" --path ~/my-project
cd ~/my-project

anvesha status                  # print the version log (per-phase outputs)
anvesha run --phase 1           # run a phase (see caveat)
anvesha resume --phase 1        # resume after a crash, from the last checkpoint
anvesha logs --phase 1 --follow # tail a phase log
```

Long phases are foreground processes — run them inside `tmux` for connection
resilience (Anvesha does not manage tmux itself; see `docs/ANVESHA.md` §9).

### Caveat — phases do not yet run end-to-end

`anvesha init`, `status`, `logs`, and the test suite work today. **Phase 1
cannot yet fetch real papers**: the MCP transport layer that connects to the
PapersFlow and paper-search-mcp servers is not implemented. Phase logic is
complete and fully tested against the injected `MCPClient` protocol (per the
Phase 1 spec's NFR-7), but `anvesha run --phase 1` has no live data source until
that transport layer is built.

## Repository layout

```
anvesha_workspace/
├── README.md                 # this file
├── docs/                     # design specs (charter + per-phase)
└── anvesha/                  # the Python package (pip-installable)
    ├── pyproject.toml
    ├── anvesha/              # source: core/, workspace/, phases/, cli.py
    └── tests/
```

## License

Not yet specified.
