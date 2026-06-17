# Anvesha — Research Assistant Kit for ML and CV Research

> **Anvesha** (अन्वेषा) — Sanskrit for *the seeking of knowledge*. The tool
> accompanies the researcher through systematic inquiry — handling the
> mechanics of literature discovery, gap analysis, planning, and
> experimentation — so the researcher's time is spent on judgment, design,
> and insight rather than orchestration.

This document is the **project charter** for Anvesha. It is the first
document Claude Code should read when working on any part of the system. It
establishes what Anvesha is, how it is structured, what conventions apply
everywhere, and how the per-phase implementation documents fit together.

For phase-level implementation details, see the per-phase documents (e.g.
`research_gap_pipeline_implementation.md`).

---

## 1. What Anvesha Is

Anvesha is a phase-driven research SDK that automates the mechanics of a
deep learning research workflow end-to-end. It is shipped as a single PyPI
package and exposes a command-line interface. Each phase of a research
project — literature survey, gap analysis, base paper selection, idea
generation, experiment planning, implementation, training, evaluation,
error analysis, future ideation, and loop decision — runs as a discrete
Anvesha command and produces a structured Markdown file in the researcher's
project workspace.

The researcher always remains in control. Anvesha handles the orchestration,
the LLM calls, the multi-agent review where it adds value, the GPU
scheduling, the checkpoint/resume on long runs, and the file I/O. The
researcher decides which phase to run, reviews each output MD before
proceeding, and makes all judgment calls that affect research direction
(e.g. base paper selection).

**Target audience.** Anyone doing serious research in machine learning or
computer vision — PhD students, postdocs, industry research engineers,
independent researchers, and master's students working on a thesis or
research project. The pipeline assumes the user has a GPU box available
for training and is comfortable with command-line tools.

---

## 2. The Problem Anvesha Solves

Research workflows are fragmented, hard to reproduce, and lose context
across iterations. Each phase of a project lives in a different surface — a
notebook here, a Notion doc there, scattered prompts to a chatbot, results
buried in a tmux scrollback. When an experiment fails (most do), the
researcher cannot easily trace which decisions led where. When a new idea
emerges, prior context is hard to reload.

Anvesha addresses this by enforcing four properties across every research
project:

1. **Phase isolation.** Each phase has a single defined input, a single
   defined output Markdown file, and a strict whitelist of tools it may use.
   No phase reaches across boundaries.
2. **Traceability.** Every output MD is versioned. When a phase is re-run
   after looping back, the old version is preserved (`03_research_gaps.md`
   becomes `03_research_gaps_v2.md`, never overwritten).
3. **Reproducibility.** The output of any phase, plus the project config,
   is sufficient to reproduce every downstream phase.
4. **Local-first execution.** The default LLM backend is self-hosted on the
   researcher's own GPU. Cloud APIs exist as fallback, not as the primary path.

---

## 3. The Eleven Phases

Each phase produces exactly one Markdown file in the researcher's project
workspace under `phases/`. The next phase reads only the prior MD files
listed as its inputs, never the conversation history that produced them.

| # | Phase | Input(s) | Output |
|---|---|---|---|
| 1 | Filter Literature | Topic statement + filter config (conferences, domains, years) | `filtered_literature/` directory — index `01_filtered_literature.md` + downloaded `pdfs/` |
| 2 | Literature Survey | `filtered_literature/` (index + PDFs) | `02_literature_survey.md` |
| 3 | Research Gaps | `02_literature_survey.md`, optionally `03_approach.md` | `03_research_gaps.md` |
| 4 | Base Paper Selection | `02`, `03` | `04_base_paper.md` |
| 5 | Idea Generation | `03`, `04` | `05_ideas.md` |
| 6 | Experiment Planning | `04`, `05` | `06_experiment_plan.md` |
| 7 | Implementation | `06` | `07_implementation_notes.md` + Git commits to researcher's code repo |
| 8 | Training | Researcher's code (on GPU box) | `08_results.txt` |
| 9 | Evaluation & Analysis | `06`, `08` | `09_evaluation.md` |
| 10 | Future Ideation | `09` | `10_future_ideation.md` |
| 11 | Loop Decision Gate | `09`, `10` | `11_loop_decision.md` |

**Phase 1 (Filter) and Phase 2 (Survey) are distinct.** Filter does
high-volume screening — a broad search yields hundreds of papers, which
are screened against the filter config (conferences, domains, years) down
to a curated reading list, and the PDFs of retained papers are downloaded
(the PRISMA "identification + screening" stage). Filter's output is a
directory: the index `01_filtered_literature.md` plus a `pdfs/` folder.
Survey then deep-reads those PDFs and synthesizes methods, datasets,
architectures, findings, and trends into one document (the PRISMA
"eligibility + synthesis" stage). Separating them gives the researcher a
checkpoint to review the filtered list before committing to deep reading,
and documents the screening rationale for reproducibility.

**Phase 3 (Research Gaps) has two modes.** If the researcher provides
`03_approach.md` describing their current method, Phase 3 runs in
**Refinement Mode** and finds gaps between what the field knows and what
the approach does — *improvement research*. If `03_approach.md` is absent,
Phase 3 runs in **Discovery Mode** and finds open problems in the field
itself: contradictions, missing benchmarks, unexplored directions —
*original-contribution research*. The mode is detected automatically.

**Phase 9 (Evaluation & Analysis)** combines two cognitive tasks in one
structured output: **evaluation** (what happened — quantitative comparison
against success criteria) and **error analysis** (why it happened — root
causes, failure modes, unexpected observations). Both sections appear in
the same Markdown file, clearly separated, so the "why" is never skipped.

**Phase 11 (Loop Decision Gate)** produces a routing decision: which
earlier phase to re-enter based on what was learned. Looping back creates
`_v2`, `_v3` versions of all downstream files without losing the original.

For a detailed description of what each phase does, exit criteria, and
operating rules, see the corresponding `PHASE_NN_*.md` document.

---

## 4. Core Design Principles

These principles recur across every phase. They are baked into Anvesha's
architecture and should be treated as constraints during any implementation.

### 4.1 Phase isolation by single-source input

A phase reads only the files listed as its inputs. It does not access prior
conversations, prior agent transcripts, or files from unrelated phases.
This is enforced at three levels: by the phase prompt template, by the
orchestration code that restricts file access, and by the tool whitelist
that limits which MCPs the phase may call.

### 4.2 LLM-agnostic core with local-first defaults

The core is built around an `LLMClient` abstraction that any backend can
implement. The shipped backends are:

- **Local primary:** vLLM serving Nemotron 3 Super (Generator/Judge roles)
  and GPT-OSS-120B (Critic role). Ollama, llama.cpp, and HuggingFace
  in-process adapters are also supported for smaller setups.
- **Cloud fallback:** Anthropic Claude, OpenAI, Google Gemini. Used only on
  timeout, parse error, GPU OOM, or low-confidence triggers.

The default backend is **always local**. Cloud is the safety net, not the
norm. This is a deliberate design decision driven by data sovereignty,
cost predictability, and the batch-sequential nature of research workloads.

### 4.3 Sequential execution by default

The default execution mode is sequential: one LLM model loaded per GPU node
at a time, with the GPU scheduler unloading and loading models between
agent calls. This halves peak VRAM compared to parallel execution and
allows two large models (Generator and Critic, in Phase 2) to share the
same GPU nodes.

Parallel mode is supported but optional — used only when total VRAM across
nodes covers all required models simultaneously.

Why this matters: research pipelines are not latency-sensitive. A phase
that takes 30 minutes is fine if it starts before lunch. Optimising for
peak VRAM efficiency is more valuable than optimising for per-token speed.

### 4.4 Checkpoint and resume on every long phase

Every phase that runs more than a few minutes implements checkpointing.
State is persisted at configurable granularity (per-agent-call,
per-iteration, or per-gap) and stored under `.anvesha/checkpoints/` in the
project workspace. If a phase crashes — GPU OOM, network failure, MCP
timeout, killed process — `anvesha resume --phase N` loads the last
checkpoint and continues. No iteration work is lost.

### 4.5 Workspace separation

There are three distinct structures, each with a different owner:

| Structure | Owner | Purpose |
|---|---|---|
| **Anvesha SDK** | Anvesha team | Python package on PyPI; never edited by researchers |
| **Research project workspace** | Researcher | Phase outputs, config, checkpoints |
| **Researcher's code repo** | Researcher | Experiment code, training scripts; separate git repository |

The boundary rule: **Anvesha never writes into the researcher's code repo.**
Phase 8 (Training) reads the code repo path from `.anvesha/config.yaml` to
trigger training on the GPU box, but does not modify code files.

### 4.6 Multi-agent only where it genuinely helps

Most phases run as a single LLM call with autonomous MCP tool use. Modern
LLMs chain tool calls natively; no agent framework is needed for that.

A small number of phases benefit from explicit multi-agent structure:

- **Phase 3 (Research Gaps)** uses a Generator → Critic (3 personas) →
  Analyzer (rubric scoring) → Judge (with adaptive web search) loop. Gap
  quality is high-stakes; a wrong gap wastes Phases 4–11. The multi-agent
  structure catches confabulation that single-pass synthesis misses.
- **Phase 7 (Implementation)** may use a coder + reviewer + smoke-test
  loop. Single-shot code generation is unreliable enough to justify a
  short critique loop.
- **Phase 9 (Evaluation & Analysis)** may use a hypothesize → test →
  revise loop for the error-analysis half of the phase, where diagnostic
  reasoning benefits from explicit iteration.

All other phases use a single LLM call. The principle: **multi-step ≠
multi-agent**. Use the simplest structure that produces a reliable output.

### 4.7 Configuration over code

Every variable that might reasonably change — model assignments, GPU node
IDs, iteration limits, scoring thresholds, MCP tool whitelists per phase —
lives in YAML configuration files, not in Python code. The two key config
files are:

- **`.anvesha/config.yaml`** (per-project) — researcher edits this; lists
  models, GPU mode, code repo path, cloud fallback keys, checkpoint dir.
- **Internal package config files** — set defaults that are rarely
  overridden.

Code reads config; code does not declare config.

### 4.8 MCP integration for external knowledge

Anvesha uses the Model Context Protocol for any operation that requires
external data. The MCP servers vary by phase:

- **Phase 1** uses PapersFlow MCP, paper-search-mcp, and Zotero MCP for
  literature discovery and PDF download.
- **Phase 2's Judge agents** use a Web Search MCP for adaptive novelty
  verification.
- **Phases 3, 5, 6** use GitHub MCP and Hugging Face MCP for code and model
  discovery.

MCPs are configured per phase via the tool whitelist in
`.anvesha/config.yaml` and the per-phase implementation doc. The phase
itself never sees the MCP list directly — it receives a restricted toolset
via the LLM client.

---

## 5. Document Hierarchy

Anvesha's documentation lives at two levels. Claude Code should know
where to look for what.

| Document | Scope | Read when… |
|---|---|---|
| `ANVESHA.md` (this file) | Project-wide foundation, phase contracts, core abstractions, configuration | Starting any work; understanding the system |
| `PHASE_NN_*.md` | Per-phase implementation specification | Implementing a specific phase in depth |

**Per-phase implementation docs** follow the convention
`PHASE_NN_<NAME>.md` (e.g., `PHASE_03_GAPS.md`). Each one is a complete
implementation brief for one phase — agent specifications, schemas,
control flow, scoring rubrics, search strategies, configuration variables,
and implementation sequence. The Research Gaps implementation doc
(`research_gap_pipeline_implementation.md`, covering Phase 3) is the
reference example.

When Claude Code is asked to implement a feature, the input context should
be: `ANVESHA.md` + the relevant `PHASE_NN_*.md`. It should not need to
read other phase docs or unrelated source files.

---

## 6. Repository Layout (High Level)

### 6.1 Anvesha SDK (`anvesha/` — the Python package)

```
anvesha/
├── cli.py                       # entry point: init | run | resume | status | logs
├── core/                        # shared infrastructure across all phases
│   ├── adapters/                # LLM backend adapters (local + cloud)
│   ├── gpu/                     # GPU node management, load/unload scheduling
│   ├── checkpoint/              # state persistence and resume
│   ├── config/                  # YAML-driven configuration classes
│   └── hooks.py                 # pre/post agent callbacks
│
├── workspace/                   # research project I/O
│   ├── project.py               # locates and validates project root
│   ├── phase_io.py              # all phase MD reads and writes (handles versioning)
│   └── version_log.py           # _VERSION_LOG.md management
│
└── phases/
    ├── phase01_filter/         # one module per phase
    ├── phase02_survey/
    ├── phase03_gaps/           # ← Research Gaps (see research_gap_pipeline_implementation.md)
    ├── phase04_base_paper/
    ├── ...
    └── phase11_loop_gate/
```

For the detailed module structure of any phase, see its per-phase
implementation doc.

### 6.2 Research Project Workspace (`research_project/`)

The researcher's project directory. Anvesha reads from and writes to this
structure; the researcher owns it.

```
research_project/
├── README.md                    # project overview (researcher-maintained)
├── _VERSION_LOG.md              # Anvesha writes after each phase completes
│
├── .anvesha/                    # gitignored — Anvesha-managed
│   ├── config.yaml              # models, GPU, paths, code repo location
│   └── checkpoints/             # phase run checkpoints for resume
│
├── filtered_literature_v1_.../  # Phase 1 output — a versioned DIRECTORY (§4.1 of the
│   ├── 01_filtered_literature.md#   Filter spec): the paper index ...
│   └── pdfs/                     #   ... plus downloaded PDFs (paper_NNN.pdf)
│
├── phases/                      # single-file outputs for Phases 2–11
│   ├── 02_literature_survey.md  # Phase 2 reads the filtered_literature/ dir above
│   ├── 03_approach.md           # optional — switches Phase 3 to Refinement Mode
│   ├── 03_research_gaps.md
│   ├── ... (through 11)
│   └── (versioned re-runs: 05_ideas_v2.md, etc.)
│
├── data/                        # researcher-owned datasets (gitignored)
├── refs/                        # researcher-owned PDFs and BibTeX
└── runs/                        # training logs pulled from GPU box (Phase 8)
```

Phase 1 is the one phase whose output is a **directory** rather than a single
file, because it bundles the index with the downloaded PDFs. Its directory
name encodes version/domain/years (e.g. `filtered_literature_v1_llm_agents_2023_2025/`),
and `_VERSION_LOG.md` records which directory is active. Every other phase
(2–11) writes a single Markdown file into `phases/`.

### 6.3 Researcher's Code Repository (separate)

The researcher's experiment code lives in a completely separate git
repository, on its own remote, with its own lifecycle. Anvesha references
it via a path in `.anvesha/config.yaml` and triggers training jobs on the
GPU box from it, but never modifies its files.

---

## 7. Core Abstractions

Two abstractions sit at the heart of Anvesha's design: the `Phase` base
class (which every phase module extends) and the `LLMClient` interface
(which every LLM backend implements). Together they define the contracts
that make the system modular, swappable, and uniformly testable.

### 7.1 The `Phase` Base Class

The central abstraction of the pipeline. Every phase has the same shape:
declare inputs, declare permitted tools, build a prompt, call the LLM,
validate the output, commit to disk. The base class implements this
lifecycle; subclasses customize only the task-specific parts.

**Responsibility.** Owns the lifecycle of one phase execution:

1. Pre-flight check that all declared input files exist
2. Build a phase prompt using a strict template (operating rules, inputs
   list, output path, tool whitelist, task description)
3. Restrict the MCP toolset to only what is permitted for this phase
4. Call the injected LLM backend with the prompt and restricted tools
5. Validate the produced Markdown against phase-specific structural
   requirements
6. Resolve the output path (with auto-versioning if the file already
   exists)
7. Write to disk and record in the version log

**Constructor inputs.**

- `project_root` — absolute path to the research project directory
- `version_log` — the `VersionLog` instance for this project
- `llm` — any `LLMClient` subclass; the phase is backend-agnostic
- `spec` — the `PhaseSpec` data for this phase (see below)

**`PhaseSpec` data.** A dataclass holding the declarative config for a
phase, loaded from `.anvesha/config.yaml`:

- `phase_id` (int) — 1 through 11
- `name` (str) — human-readable name
- `inputs` (list of paths) — files the LLM may read
- `output` (str) — path of the Markdown file to produce
- `tools_permitted` (list of MCP tool names) — strict whitelist
- `template` (str) — Jinja2 template name for the output structure
- `exit_criteria` (list of validator names) — checks to run on output

**Subclasses must implement.**

- `task_description() → str` — the phase-specific paragraph that goes into
  the prompt
- `validate_output(content: str) → ValidationResult` — phase-specific
  structural checks (e.g. "must have ≥ 20 rows in Paper Inventory table")

**Subclasses inherit.**

- `validate_inputs()` — checks declared input files exist
- `build_prompt()` — assembles the standard prompt block (see below)
- `run()` — orchestrates the full lifecycle
- `_resolve_output_path()` — handles `_v2`, `_v3` versioning automatically

**Phase prompt template.** Every phase prompt follows this exact
structure. The base class assembles it; subclasses fill in only the task
description.

```
PHASE: <N> — <name>

OPERATING RULES
- Read ONLY the files listed in INPUTS.
- Produce EXACTLY ONE output file at the path in OUTPUT.
- Follow the output template precisely.
- Do not start, plan, or hint at the next phase.
- If INPUTS are missing or incomplete, stop and ask.

INPUTS
- <input_file_1>
- <input_file_2>

OUTPUT
- <output_path>

TOOLS PERMITTED
- <tool_1>
- <tool_2>

TASK
<task_description from subclass>
```

This template is what enforces phase isolation. It is not optional, not
customizable per phase, and not bypassable.

### 7.2 The `LLMClient` Interface

Every LLM backend (vLLM, Nemotron, Ollama, Anthropic, OpenAI, Google,
FallbackAdapter) implements the same minimal interface. Phases never know
which backend is in use.

**Interface.** A single method:

- `run(prompt: str, mcps: list, input_files: list[Path]) → LLMResponse`

**The implementation must:**

1. Inject the contents of `input_files` into the prompt context
2. Convert the list of MCP clients into the backend's native tool format
3. Run a tool-call loop: send prompt → if the model emits tool calls,
   dispatch them via the MCP clients, append results, repeat
4. Return when the model produces a final text response

**`LLMResponse` data.**

- `content` (str) — the produced Markdown text
- `tool_calls_made` (int) — for diagnostics
- `tokens_in` (int) — for cost and capacity tracking
- `tokens_out` (int) — for cost and capacity tracking

**Concrete backends.** See Section 10 (Tech Stack) for the full list of
shipped backends. The pattern across all of them: same interface, same
return type, different transport.

**Backend selection.** Resolved at CLI time by a factory function with
this precedence:

1. `--backend` CLI flag — one-off override
2. `phase_overrides[N]` in `.anvesha/config.yaml` — per-phase preference
3. `default_backend` in `.anvesha/config.yaml` — project default

The phase itself never sees this resolution. It receives an `LLMClient`
already constructed and ready to call.

### 7.3 Relationship Between `phases/` and `core/`

Phase modules under `anvesha/phases/` are thin. They declare what the
phase does (`task_description`), what its output must look like
(`validate_output`), and — for multi-agent phases like Phase 3 — what
internal coordination logic applies. Everything else comes from `core/`:
LLM adapters, GPU scheduling, checkpointing, configuration loading, the
MCP registry, the workspace I/O.

This is the single most important architectural rule in Anvesha. **Phase
modules do not implement infrastructure.** If a phase finds itself reaching
into HTTP clients, GPU management, or file I/O, that logic belongs in
`core/`, not in the phase. The phase calls `self.llm.run(...)`,
`self.workspace.write(...)`, `self.checkpoint.save(...)` — the rest of its
code is research logic, not plumbing.

---

## 8. Project Configuration

Every Anvesha project has a single configuration file at
`.anvesha/config.yaml` in the project workspace root. This file is the
only place a researcher edits to customize Anvesha's behavior for their
project. Every phase reads from it; nothing in a phase module is
hardcoded that could reasonably vary per project.

What goes in the config:

- **Project metadata** — workspace root, researcher's code repo path
- **Phase file paths and inputs** — which Markdown file each phase reads
  and writes
- **Tool whitelists per phase** — which MCPs each phase may call
- **LLM backends and assignments** — local servers, cloud fallbacks,
  per-phase or per-agent-role overrides
- **GPU and execution mode** — node IDs, sequential vs parallel
- **Checkpoint settings** — granularity, storage, directory
- **Phase 8 training remote** — SSH connection details for the GPU box

What does **not** go in the config:

- Agent prompts, scoring rubrics, output template structure — these live
  in the package and per-phase docs
- Phase orchestration logic — fixed in the `Phase` base class
- The list of phases — the eleven phases are fixed

### 8.1 Configuration Precedence

Settings resolve in this order. Later sources override earlier ones.

1. **Built-in package defaults** — ship with Anvesha; cover every field
2. **`.anvesha/config.yaml`** — per-project overrides
3. **Environment variables** — for sensitive values (API keys, SSH paths)
4. **CLI flags** — one-off overrides for a specific invocation

A minimal config file only needs to specify what differs from defaults.
Many projects need only the project metadata block.

### 8.2 Minimal Configuration

The smallest viable `.anvesha/config.yaml`:

```yaml
project:
  name: "my-research-project"
  code_repo: "/home/researcher/my-cv-code"

llm:
  default_backend: "vllm_nemotron"
  backends:
    vllm_nemotron:
      type: "vllm"
      base_url: "http://localhost:8000/v1"
      model: "nvidia/nemotron-3-super-120b-a12b"
```

Everything else falls back to package defaults: standard phase file
paths, default tool whitelists, sequential execution, filesystem
checkpoints.

### 8.3 Full Configuration Structure

The config file has six top-level sections. Below is the shape of each;
field-level details live in the per-phase implementation docs.

```yaml
# === Project metadata ===
project:
  name: "..."                         # project identifier
  code_repo: "/path/to/code/repo"     # researcher's separate code repo

# === LLM configuration ===
llm:
  default_backend: "..."              # name of a backend defined below
  backends:
    <backend_name>:                   # one block per backend
      type: "vllm" | "ollama" | "anthropic" | "openai" | "google" | ...
      # backend-specific fields
  phase_overrides:                    # optional per-phase backend override
    2: "<backend_name>"
    3: "<backend_name>"
  role_assignments:                   # for multi-agent phases (Phase 3 etc.)
    phase_03:
      generator: "<backend_name>"
      critic:    "<backend_name>"
      analyzer:  "<backend_name>"
      judge:     "<backend_name>"

# === Per-phase paths and tools ===
phases:
  1:
    inputs:
      - "README.md"                   # topic statement
    output: "filtered_literature/01_filtered_literature.md"   # written into a versioned dir
    tools:
      - "papersflow.*"                # discovery + metadata
      - "paper-search.search_papers"  # discovery
      - "paper-search.download_with_fallback"   # PDF acquisition
      - "filesystem.*"
    filters:                          # full schema in filter_literature_implementation.md §4
      conferences: ["NeurIPS", "ICML", "CVPR"]
      domains: ["LLM Agents", "Retrieval-Augmented Generation"]
      years: { start: 2023, end: 2025 }
    options:
      max_papers: 40
      relevance_threshold: 0.50
      pdf_download: true
  2:
    inputs:
      - "filtered_literature/01_filtered_literature.md"   # the Phase 1 dir (index + pdfs/)
    output: "phases/02_literature_survey.md"
    tools:
      - "filesystem.*"                # reads the already-downloaded PDFs + index; writes the survey
  3:
    inputs:
      - "phases/02_literature_survey.md"
      - "phases/03_approach.md"       # optional; if present, Refinement Mode
    output: "phases/03_research_gaps.md"
    tools:
      - "papersflow.verify_citation"
      - "papersflow.expand_citation_graph"
      - "web_search.*"
      - "filesystem.*"
  # ... phases 4 through 11

# === GPU and execution mode ===
gpu:
  mode: "auto" | "manual" | "single"
  execution_mode: "sequential" | "parallel"
  available_node_range: [0, 7]
  vram_headroom_gb: 8.0

# === Checkpoint ===
checkpoint:
  enabled: true
  granularity: "agent" | "iteration" | "gap"
  storage: "filesystem" | "sqlite"
  dir: ".anvesha/checkpoints"

# === Phase 8 training remote (only if running training via Anvesha) ===
training:
  gpu_box:
    host: "user@gpu-server.example.edu"
    ssh_key: "~/.ssh/id_rsa"
  remote_workdir: "/home/user/training_runs"
```

### 8.4 Default Phase Paths

The package ships with these defaults. Override only if your project uses
a different filename convention.

| Phase | Default input(s) | Default output |
|---|---|---|
| 1 | `README.md` (topic statement) + `filters` config | `filtered_literature/01_filtered_literature.md` (+ `pdfs/`) in a versioned directory |
| 2 | `filtered_literature/01_filtered_literature.md` (index + `pdfs/`) | `phases/02_literature_survey.md` |
| 3 | `phases/02_literature_survey.md` (+ optional `phases/03_approach.md`) | `phases/03_research_gaps.md` |
| 4 | `phases/02_literature_survey.md`, `phases/03_research_gaps.md` | `phases/04_base_paper.md` |
| 5 | `phases/03_research_gaps.md`, `phases/04_base_paper.md` | `phases/05_ideas.md` |
| 6 | `phases/04_base_paper.md`, `phases/05_ideas.md` | `phases/06_experiment_plan.md` |
| 7 | `phases/06_experiment_plan.md` | `phases/07_implementation_notes.md` |
| 8 | researcher's code repo (on GPU box) | `phases/08_results.txt` |
| 9 | `phases/06_experiment_plan.md`, `phases/08_results.txt` | `phases/09_evaluation.md` |
| 10 | `phases/09_evaluation.md` | `phases/10_future_ideation.md` |
| 11 | `phases/09_evaluation.md`, `phases/10_future_ideation.md` | `phases/11_loop_decision.md` |

### 8.5 Versioned Re-runs and Config

When a phase is re-run (after looping back), Anvesha auto-versions the
output file (`03_research_gaps.md` → `03_research_gaps_v2.md`). The
config file does not need updating to reflect this — the path resolver
in the `Phase` base class handles it automatically by checking what
already exists on disk.

Downstream phases reading the output use the current version recorded in
`_VERSION_LOG.md`, not the base path in config. Config defines the
*default* path; the version log defines the *current* path.

### 8.6 Where Configuration is Loaded

Configuration is loaded once at the start of each CLI invocation by the
`core/config/` module. The loaded `PipelineConfig` object is then
injected into the `Phase` instance (alongside `LLMClient`, `VersionLog`,
and `WorkspaceProject`). No phase ever reads `.anvesha/config.yaml`
directly — it receives only the resolved configuration values it needs.

---

## 9. CLI Surface

Anvesha is invoked exclusively through the `anvesha` command. There is no
Python API intended for direct researcher use — the CLI is the supported
interface.

| Command | Purpose |
|---|---|
| `anvesha init --name <project>` | Create a new research project workspace |
| `anvesha run --phase N` | Run phase N using the default backend |
| `anvesha run --phase N --backend <name>` | Run phase N with a specific backend override |
| `anvesha resume --phase N` | Resume phase N from last checkpoint after crash |
| `anvesha status` | Print `_VERSION_LOG.md` for the current project |
| `anvesha logs --phase N --follow` | Tail the log file for a running or completed phase |

Anvesha runs as a foreground process. For long phases (especially Phase 3,
Research Gaps), the researcher is expected to invoke it inside `tmux` for
connection resilience. Anvesha does not depend on tmux, manage tmux
sessions, or detach itself.

---

## 10. Tech Stack

### 10.1 Primary local LLMs (default backends)

- **Generator / Judge roles:** NVIDIA Nemotron 3 Super (120B-A12B MoE),
  served via vLLM or TensorRT-LLM. 1M token context window. Reasoning
  level defaults to `"high"`.
- **Critic roles:** OpenAI GPT-OSS-120B, served via vLLM with Harmony
  Format enabled. Reasoning effort defaults to `"high"`.
- **Analyzer / Merger:** any capable model; defaults match the Generator
  model. Both are candidates for `"low"` reasoning if latency matters.

The Generator/Critic split uses different model families deliberately —
correlated blind spots are the most common failure mode in single-family
multi-agent setups.

### 10.2 Fallback LLMs (cloud)

- Anthropic Claude (Opus or Sonnet)
- OpenAI GPT
- Google Gemini

Cloud is invoked only on failure triggers (timeout, parse error, GPU OOM,
low confidence) defined per-adapter in the fallback configuration.

### 10.3 LLM serving stacks supported

- vLLM (primary — supports both Nemotron and GPT-OSS)
- TensorRT-LLM (alternative for Nemotron)
- Ollama (lighter setups, OpenAI-compatible API)
- llama.cpp server mode
- HuggingFace transformers (in-process, no server)

### 10.4 MCP servers used across phases

- **Web search MCP** — Phase 3 Judge agents (adaptive search loop)
- **PapersFlow MCP** — Phase 1 (broad search, citation graphs), Phase 3 (citation verification)
- **paper-search-mcp** — Phase 1 (search), Phase 2 (PDF download across 14 sources)
- **GitHub MCP** — Phase 4, 6, 7 (code repo inspection)
- **Hugging Face MCP** — Phase 4, 6 (models, datasets, Spaces)
- **Zotero MCP** — Phase 1, 2, 4 (researcher's personal library)
- **Filesystem MCP** — every phase (Anvesha-internal file I/O)

### 10.5 Python ecosystem

- Python 3.10+
- `pydantic` for schemas and validation
- `pyyaml` for config
- `openai` library (for OpenAI-compatible servers including vLLM and Ollama)
- `anthropic`, `google-generativeai` for cloud fallbacks
- `paramiko` or `fabric` for SSH to GPU box (Phase 8)
- `sqlite3` (optional, for checkpoint storage)

---

## 11. Versioning Convention

When a phase runs and its output file already exists, Anvesha writes the
new output to a `_v2`, `_v3`, ... suffixed path. The original is preserved.

```
First run:   phases/03_research_gaps.md
Re-run:      phases/03_research_gaps_v2.md
Third run:   phases/03_research_gaps_v3.md
```

`_VERSION_LOG.md` tracks the current version of every phase. Looping back
to phase N marks phases N through 11 as needing regeneration; their next
run produces `_v2` (or higher) versions automatically.

This is automatic. The researcher does not pass a version flag — the path
resolver handles versioning by checking what already exists.

**Phase 1 is the exception: it versions at the directory level.** Because
Phase 1's output is a directory (the index plus `pdfs/`), re-running it
produces a new versioned directory (`filtered_literature_v1_...`,
`filtered_literature_v2_...`) rather than a `_v2`-suffixed file. The
directory name encodes version, domain, and year range, and `_VERSION_LOG.md`
records which directory is active. Phases 2–11 version at the file level as
described above.

---

## 12. Boundary Rules (Hard Constraints)

These rules apply across the entire system. Violating them produces bugs
that are difficult to diagnose.

1. **Phase modules do not import from other phase modules.** Phase outputs
   travel via workspace Markdown files, never via in-memory objects.
2. **Core is phase-agnostic.** Nothing under `core/` knows about specific
   phases. Adapters, GPU management, checkpointing, and config are
   identical across all phases.
3. **The workspace module owns all file I/O.** No agent, loop, or phase
   module reads or writes files directly. All file access flows through
   `workspace/phase_io.py`. Agents receive Markdown content as strings,
   not file paths.
4. **Anvesha never writes into the researcher's code repo.** The code repo
   is researcher-owned and managed by their own git workflow. Phase 8
   (Training) reads the repo path to trigger runs; it does not modify files.
5. **`.anvesha/` is always gitignored in the research project.**
   Checkpoints and run state are machine-local, not version-controlled.
6. **Phase entry points accept a `ResearchProject` object, not raw paths.**
   `ResearchProject` (in `workspace/project.py`) resolves all file paths
   from the project root. Hardcoded paths anywhere in a phase module are
   a bug.
7. **Phase 8 (Training) is always a thin remote wrapper, never a trainer.**
   Training code lives in the researcher's code repo. Phase 8 only handles
   SSH, job submission, log pulling, and result writing.
8. **Tool whitelists are enforced per phase.** A phase cannot call an MCP
   tool that is not in its declared whitelist. This is checked at
   orchestration time, before the LLM is invoked.

---

## 13. Implementation Status

Anvesha is built phase by phase. Each phase has a per-phase implementation
doc and is implemented end-to-end before the next begins.

| Phase | Implementation doc | Status |
|---|---|---|
| 1 — Filter Literature | (pending) | Not started |
| 2 — Literature Survey | (pending) | Not started |
| 3 — Research Gaps | `research_gap_pipeline_implementation.md` | Implementation spec complete |
| 4 — Base Paper Selection | Embedded in `research_gap_pipeline_implementation.md` (Section 5.8) | Spec complete |
| 5 — Idea Generation | (pending) | Not started |
| 6 — Experiment Planning | (pending) | Not started |
| 7 — Implementation | (pending) | Not started |
| 8 — Training | (pending) | Not started |
| 9 — Evaluation & Analysis | (pending) | Not started |
| 10 — Future Ideation | (pending) | Not started |
| 11 — Loop Decision Gate | (pending) | Not started |

**Shared infrastructure** (the `core/` and `workspace/` modules) should be
built first, before any phase-specific code, because every phase depends on
it. The Research Gaps implementation doc
(`research_gap_pipeline_implementation.md`) Section 9 ("Implementation
Sequence") gives the recommended build order including the infrastructure
stages.

---

## 14. How Claude Code Should Use These Documents

### 14.1 Reading order for any task

1. **Always start here** (`ANVESHA.md`) — to understand the system. Pay
   particular attention to Sections 7 (Core Abstractions), 8 (Project
   Configuration), and 12 (Boundary Rules) if the task involves `core/`,
   `workspace/`, or cross-phase code.
2. **Then the relevant `PHASE_NN_*.md`** — if implementing or modifying a
   specific phase.

Do not read unrelated per-phase docs. They are large and will dilute
context.

### 14.2 When implementing a new phase

1. Confirm `ANVESHA.md` is in context.
2. Confirm the per-phase implementation doc (`PHASE_NN_*.md`) is in
   context.
3. Follow the implementation sequence given in the per-phase doc, not a
   self-generated plan.
4. Respect the boundary rules in Section 12 absolutely. They are
   non-negotiable.
5. New configuration variables go in YAML (project config or package
   defaults), not in Python.

### 14.3 When modifying shared infrastructure

1. Confirm `ANVESHA.md` is in context, especially Section 7 (Core
   Abstractions) and Section 12 (Boundary Rules).
2. Identify which phases depend on the component being modified (this is
   usually "all of them").
3. Changes to `LLMClient`, `Phase` base class, `PhaseSpec`, the workspace
   module, or the checkpoint format are breaking changes — require
   updating the affected phase docs.

### 14.4 What never to do

- Do not invent new phases. The eleven phases are fixed; new functionality
  goes into an existing phase or into shared infrastructure.
- Do not bypass the workspace module for file I/O.
- Do not import across phase boundaries.
- Do not introduce a new LLM backend without adding it as a proper
  `LLMClient` subclass under `core/adapters/`.
- Do not introduce a new MCP without declaring it in `mcp_registry` and
  whitelisting it per phase.
- Do not write training code into Anvesha. That lives in the researcher's
  code repo.

---

## 15. The Name and Its Meaning

**Anvesha** (अन्वेषा) comes from the Sanskrit root *anviṣ* — "to seek, to
search, to investigate." It means "the seeking of knowledge" and is also
used as a given name in India for someone who seeks understanding.

The name was chosen for three reasons:

1. **The meaning is exact.** Anvesha is what the tool does — it
   systematically seeks knowledge alongside the researcher.
2. **It is humble.** The name positions Anvesha as a *seeker*, not an
   autonomous solver. The researcher does the research; Anvesha
   accompanies and assists.
3. **It is global.** Three syllables, soft consonants, pronounceable in
   any language, distinctive in the ML tooling landscape.

> *Anvesha — a methodos for systematic inquiry.*
