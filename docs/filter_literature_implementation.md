# Filter Literature — Technical Implementation Specification
## Phase 1 of the Anvesha Pipeline · v2.0

> **Audience:** (1) engineers implementing the feature, and (2) downstream LLM workflows that consume its outputs.
>
> **Scope:** This document is a complete, prescriptive implementation specification for the **Filter Literature** feature (Phase 1 of Anvesha). It defines functional and non-functional requirements, input/output schemas, the processing workflow, and every algorithm and rule needed to build the feature deterministically.
>
> **Relationship to Anvesha:** Phase 1 is the entry point of the eleven-phase Anvesha pipeline. Its primary artifact, `01_filtered_literature.md`, is consumed by Phase 2 (Literature Survey), which deep-reads the curated papers. See `ANVESHA.md` for the project-wide architecture, the `Phase` base class, configuration system, and shared infrastructure (LLM adapters, checkpointing). This spec does not repeat that material; it references it where relevant.
>
> **Data sources (mandatory):** **PapersFlow** and **paper-search-mcp**. PapersFlow handles discovery and metadata; paper-search-mcp handles search and **PDF acquisition**. No other discovery or repository-search tool is permitted (see §8).

---

## Table of Contents

1. Functional Requirements
2. Non-Functional Requirements
3. Input Schema
4. Output Schema
5. Processing Workflow
6. Ranking Strategy
7. Deduplication Strategy
8. Repository Detection Logic
9. PDF Management Workflow
10. Error Handling
11. Example Configuration
12. Example Output
13. Future Extensibility Considerations

Appendix A — Architecture Diagram
Appendix B — Internal Type Definitions
Appendix C — SDK Package Structure
Appendix D — Shared Infrastructure (reference)

---

## 1. Functional Requirements

The feature filters academic literature against user-defined criteria, retrieves matching papers, produces a structured machine-consumable index, and acquires supporting artifacts (PDFs) for downstream pipelines.

**FR-1 — Filter-driven retrieval.** The feature MUST retrieve papers matching a filter configuration consisting of target conferences, research domains, and a publication-year range (§3).

**FR-2 — Multi-source discovery.** Discovery and metadata retrieval MUST use PapersFlow and paper-search-mcp. Results from both MUST be aggregated.

**FR-3 — Relevance ranking.** Retained papers MUST be ranked by relevance to the configured filters, using a deterministic scoring and sorting procedure with explicit tie-breaking (§6).

**FR-4 — Deduplication.** Papers MUST be deduplicated. Title similarity is the required deduplication signal, with exact identifier (DOI / arXiv ID) match as a stronger precondition (§7).

**FR-5 — PDF acquisition.** For every retained paper, the feature MUST attempt to download and store a PDF via paper-search-mcp, under a deterministic naming scheme in a `pdfs/` subdirectory (§9).

**FR-6 — Per-paper summaries.** For every retained paper, the feature MUST generate an LLM-written, implementation-focused summary (§5, Stage 8).

**FR-7 — Repository detection (metadata-only).** The feature MUST set `git_exists` and `code_url` for every retained paper, derived **exclusively** from explicit code links present in the paper's metadata. The feature MUST NOT perform GitHub searches, web searches, or any repository discovery outside the metadata returned by the discovery tools (§8).

**FR-8 — Structured Markdown output.** The feature MUST produce `01_filtered_literature.md`, a metadata-first, machine-consumable document with one YAML record per paper followed by its summary (§4).

**FR-9 — Versioned output directories.** The feature MUST write its outputs into a directory whose name MAY encode version, domain, and year range (e.g. `filtered_literature_v1_llm_agents_2023_2025/`). Re-running with different criteria MUST NOT overwrite a prior run's directory (§4.1).

**FR-10 — Provenance and accounting.** The output MUST record the filter configuration used and a PRISMA-style funnel (identified → deduplicated → hard-filtered → screened → retained) so the result set is reproducible and auditable (§4.2).

**FR-11 — Graceful empty result.** If filtering eliminates all candidates, the feature MUST NOT crash. It MUST emit a diagnostic output documenting the funnel and recommending which filter to relax (§10, EH-7).

---

## 2. Non-Functional Requirements

**NFR-1 — Determinism.** Given identical inputs (filter config, tool responses, model, seed), the feature MUST produce byte-identical `01_filtered_literature.md` output. To achieve this:
- LLM scoring and summary calls use greedy / temperature-0 decoding.
- Sorting is fully ordered via the tie-break chain in §6.3.
- YAML fields are emitted in a fixed canonical order (§4.3).
- Collections (authors, domains) preserve source order; sets are sorted lexicographically before emission.

**NFR-2 — Machine-first formatting.** Output is optimized for LLM/program consumption, not humans:
- Every field is explicitly present; absent values are the literal `null`, never omitted.
- No prose outside the designated `## LLM Summary` blocks and the run manifest.
- A single, documented parsing grammar (§4.4) governs the whole file.

**NFR-3 — Reproducibility.** The run manifest records the exact filter config, schema version, generator version, and timestamp. A second run with the same config and tool state reproduces the same retained set and ordering.

**NFR-4 — Idempotency.** Re-running with an unchanged config either (a) reuses the existing versioned directory if `overwrite=false` and content is unchanged, or (b) creates the next `_v{N+1}` directory. The feature never partially overwrites an existing directory in place.

**NFR-5 — Fault isolation.** Per-paper failures (PDF download, summary generation) are non-fatal: the paper remains in the output with explicit failure markers. Only configuration errors and total source failure are fatal.

**NFR-6 — Performance.** Searches across sources run concurrently. PDF downloads and summary generations run with bounded concurrency. LLM screening/summarization is batched where the backend supports it. The phase is light on GPU relative to synthesis phases (see Appendix D).

**NFR-7 — Boundary compliance.** The feature obeys the Anvesha boundary rules (ANVESHA.md §12): no imports from other phase modules; all file I/O via the workspace module; all LLM and MCP access via injected clients; the entry point accepts a `ResearchProject`, never raw paths.

**NFR-8 — Auditability.** Every retained paper carries its provenance (`source`), its identifiers (`doi`, `arxiv_id`), its rank, and its relevance score. Every exclusion is countable from the funnel.

---

## 3. Input Schema

### 3.1 Required Filter Configuration

The feature is driven by a filter block. This is the canonical required schema:

```yaml
conferences:
  - NeurIPS
  - ICML
  - CVPR

domains:
  - LLM Agents
  - Retrieval-Augmented Generation

years:
  start: 2023
  end: 2025
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `conferences` | list[string] | yes | Target conferences. **Hard filter** — a paper's venue must resolve to one of these (venue canonicalization, §7.4 / Appendix B). An empty list disables venue filtering. |
| `domains` | list[string] | yes | Research domains / categories / topics. **Soft signal** — drives search query generation and LLM relevance scoring. Not a hard filter. |
| `years.start` | integer | yes | Inclusive lower bound on publication year. **Hard filter.** |
| `years.end` | integer | yes | Inclusive upper bound on publication year. **Hard filter.** |

### 3.2 Optional Operational Parameters

These tune behavior; all have defaults so the required block alone is sufficient to run.

```yaml
options:
  max_papers: 40                 # cap on retained papers (PDFs+summaries only for these)
  relevance_threshold: 0.50      # drop papers scoring below this (0.0–1.0); null = keep all
  dedup_threshold: 0.90          # title-similarity threshold for duplicates (§7)
  max_results_per_source: 100    # cap on raw results pulled per source per query
  max_queries: 6                 # number of search query variations to generate
  include_preprints: true        # keep arXiv preprints whose venue cannot be resolved
  pdf_download: true             # if false, skip PDF acquisition (metadata-only run)
  pdf_max_retries: 3             # download retry attempts per paper
  summary_max_retries: 2         # LLM summary retry attempts per paper
  output_root: "."               # parent directory for the versioned output directory
  output_version: 1              # the {N} in _v{N}; auto-increments if dir exists
  dirname_include_domain: true   # include domain slug in the output directory name
  dirname_include_years: true    # include year range in the output directory name
  language: "en"                 # ISO 639-1; drop papers whose language is known and differs
```

### 3.3 Topic Statement (optional, sharpens relevance)

In the Anvesha workspace, the researcher's free-text topic statement (from `README.md`) MAY be provided alongside the filters. When present it sharpens query generation and relevance scoring. When absent, the `domains` list is the sole topical signal.

### 3.4 Hard vs Soft Filters (summary)

| Criterion | Class | Enforced by |
|---|---|---|
| `conferences` | Hard | Hard Filter Gate (§5, Stage 4) |
| `years` | Hard | Hard Filter Gate |
| `language` | Hard | Hard Filter Gate |
| `domains` | Soft | Query Builder + Relevance Scorer |
| `relevance_threshold` | Soft cutoff | Relevance Scorer |
| `max_papers`, `dedup_threshold`, caps | Control | Orchestrator / Deduplicator |

---

## 4. Output Schema

### 4.1 Output Directory Structure

The feature writes a self-contained output directory containing the index file and the PDFs:

```
filtered_literature/
├── 01_filtered_literature.md
└── pdfs/
```

The directory name MAY encode version, domain, and year range. The canonical naming pattern:

```
filtered_literature[_v{N}][_{domain_slug}][_{year_start}_{year_end}]/
```

Example:

```
filtered_literature_v1_llm_agents_2023_2025/
├── 01_filtered_literature.md
├── pdfs/
│   ├── paper_001.pdf
│   ├── paper_002.pdf
│   └── ...
```

**Directory-name construction rules (deterministic):**
1. Base segment: `filtered_literature`.
2. Version segment: `_v{N}` where `N = options.output_version`. If the resulting directory already exists and `overwrite=false`, increment `N` until an unused name is found.
3. Domain segment (if `dirname_include_domain`): `_{domain_slug}`, where `domain_slug` is the slug of the **first** domain in `domains` (lowercase; non-alphanumeric runs collapsed to single `_`; leading/trailing `_` stripped). Example: `"LLM Agents"` → `llm_agents`.
4. Year segment (if `dirname_include_years`): `_{years.start}_{years.end}`. Example: `_2023_2025`.

`output_root` (default `.`) is the parent directory. In the Anvesha workspace the output directory is created at the workspace root; `01_filtered_literature.md` within it is the canonical artifact consumed by Phase 2, and `_VERSION_LOG.md` records which versioned directory is active.

### 4.2 Primary Artifact: `01_filtered_literature.md`

The file is a **metadata-first, machine-consumable** document. It consists of:

1. **A run manifest** — a single YAML frontmatter block at the top of the file (document-level metadata + filter config + PRISMA counts). Identified by `document_type: filtered_literature_index` and the absence of `paper_id`.
2. **A sequence of paper records** in ranked order (rank 1 first). Each record is a YAML frontmatter block (identified by the presence of `paper_id`) immediately followed by a `## LLM Summary` heading and the summary body.

### 4.3 Canonical Field Order

**Run manifest fields (in this order):**
```
document_type        # literal: filtered_literature_index
schema_version       # literal: "2.0"
generated_at         # ISO 8601 UTC timestamp
generator            # literal: anvesha.phase01_filter
filters              # echo of the input filter config (conferences, domains, years)
counts               # PRISMA funnel (see below)
papers_total         # integer; number of paper records that follow
output_dir           # the versioned directory name
```

`counts` sub-fields (in order): `identified`, `after_deduplication`, `after_hard_filter`, `screened`, `retained`, `pdfs_downloaded`, `pdfs_failed`, `with_code`.

**Paper record fields (in this exact order — every field always present):**
```
paper_id             # "paper_{NNN}", zero-padded to 3 digits, assigned by final rank
title                # string
authors              # list[string]; source order preserved
conference           # canonical venue key (string) or null
year                 # integer or null
domain               # list[string]; the configured domains this paper matched
paper_url            # canonical landing-page URL (string) or null
pdf_path             # relative path "pdfs/paper_{NNN}.pdf" or null if not downloaded
pdf_status           # enum: "ok" | "failed" | "skipped"
code_url             # string or null (see §8)
git_exists           # boolean (see §8)
relevance_score      # float in [0.0, 1.0], 2-decimal precision
rank                 # integer (1 = most relevant)
doi                  # string or null
arxiv_id             # string or null
source               # enum: "papersflow" | "paper_search" | "merged"
```

### 4.4 Parsing Grammar (consumer contract)

Downstream consumers MUST parse the file as follows:

```
FILE            := RUN_MANIFEST  PAPER_RECORD*
RUN_MANIFEST    := "---" NEWLINE  YAML_BLOCK  "---" NEWLINE
PAPER_RECORD    := "---" NEWLINE  YAML_BLOCK  "---" NEWLINE
                   "## LLM Summary" NEWLINE
                   SUMMARY_TEXT
YAML_BLOCK      := one or more "key: value" lines (valid YAML)
SUMMARY_TEXT    := free text until the next line that is exactly "---" or EOF
```

Disambiguation: the **first** YAML block in the file is the run manifest (it contains `document_type` and no `paper_id`). Every subsequent YAML block is a paper record (it contains `paper_id`). A record's summary is the text between its closing `---` and the next opening `---` (or EOF).

### 4.5 Exit Criteria (output validity)

- Run manifest present and well-formed; `papers_total` equals the number of paper records.
- `counts` are internally consistent: `identified ≥ after_deduplication ≥ after_hard_filter ≥ screened ≥ retained`.
- Every paper record contains all fields from §4.3 (nulls where unknown).
- `rank` values are a contiguous sequence `1..retained`; records appear in ascending `rank`.
- Every `pdf_path` that is non-null points to an existing file in `pdfs/`.
- `git_exists == (code_url != null)` for every record (§8 invariant).

---

## 5. Processing Workflow

The pipeline is **linear** — a single pass through ordered stages, with no iterative agent loop. Only Stages 1, 7, and 8 invoke an LLM; the rest are deterministic. The Orchestrator drives the stages and maintains the funnel counts.

```
Stage 0  Load & validate config           (deterministic)
Stage 1  Query Builder                     (LLM)          → search queries
Stage 2  Multi-Source Search               (MCP)          → raw candidates      [identified]
Stage 3  Normalize & Deduplicate           (deterministic)→ unique candidates   [after_deduplication]
Stage 4  Hard Filter Gate                  (rule-based)   → filtered candidates [after_hard_filter]
Stage 5  Relevance Scoring                 (LLM, batched) → scored candidates   [screened]
Stage 6  Rank & Select                     (deterministic)→ retained set        [retained]
Stage 7  Repository Detection              (metadata parse, deterministic)
Stage 8  PDF Acquisition                   (MCP)          → pdf_path/pdf_status [pdfs_downloaded/failed]
Stage 9  Summary Generation                (LLM)          → per-paper summary
Stage 10 Output Assembly                   (deterministic)→ versioned dir + 01_filtered_literature.md
Stage 11 Update _VERSION_LOG.md            (deterministic)
```

**Stage 0 — Load & validate config.** Parse the filter block (§3.1) and options (§3.2). Validate: `conferences`/`domains` are lists; `years.start ≤ years.end`; thresholds in range. Fail fast with a clear message on invalid config (EH-1).

**Stage 1 — Query Builder (LLM).** Generate up to `max_queries` diverse search query strings from `domains` (and the topic statement if present). Queries target *topic*, not metadata — conference and year constraints are NOT embedded in query text; they are applied as hard filters later. Each query is 2–6 terms and meaningfully distinct (direct, synonym, sub-topic, method/dataset angles). Decoding is greedy for determinism.

**Stage 2 — Multi-Source Search (MCP).** Execute every query against PapersFlow (`search_literature`/`search`) and paper-search-mcp (`search_papers`, `search_arxiv`), pulling up to `max_results_per_source` per source per query. Searches run concurrently. Tag each result with its `source` and the query that surfaced it. `identified` = total raw results.

**Stage 3 — Normalize & Deduplicate (deterministic).** Map each source result into the internal `PaperCandidate` type (Appendix B). Then deduplicate per §7: exact DOI/arXiv match first, then title similarity ≥ `dedup_threshold`. Merge duplicate groups, preferring the published version for venue/year while retaining `arxiv_id` for PDF fallback, and unioning code links and author lists. `after_deduplication` = surviving count.

**Stage 4 — Hard Filter Gate (rule-based).** Apply hard filters in order, recording removals by reason:
1. **Conference filter** — resolve each candidate's venue to a canonical key (Appendix B venue canonicalization); keep if the key is in `conferences`. If `conferences` is empty, skip. Preprint exception: if the venue is unresolved but the paper is a preprint and `include_preprints=true`, keep it (it still faces relevance scoring).
2. **Year filter** — keep if `years.start ≤ year ≤ years.end`. Null year: keep only if preprint and `include_preprints=true`.
3. **Language filter** — drop if a known language differs from `options.language`; unknown language is kept.
`after_hard_filter` = surviving count.

**Stage 5 — Relevance Scoring (LLM, batched).** For each surviving candidate, the LLM produces a `RelevanceAssessment` (per-domain match levels + evidence quotes + topic alignment) on the title + abstract — **not** a numeric score (§6.1.1). Code then validates every evidence quote against the source text (§6.1.3) and computes `relevance_score ∈ [0.0, 1.0]` deterministically (§6.1.4), recording the matched `domain` subset. Greedy decoding. `screened` = count scored.

**Stage 6 — Rank & Select (deterministic).** Drop candidates below `relevance_threshold` (if set). Sort the remainder per §6.2–6.3. If `max_papers` is set, retain the top `max_papers`. Assign `rank` (1-based) and `paper_id = paper_{rank:03d}`. `retained` = final count. If `retained == 0`, jump to the empty-result path (EH-7).

**Stage 7 — Repository Detection (deterministic).** For each retained paper, derive `git_exists` and `code_url` from metadata only, per §8.

**Stage 8 — PDF Acquisition (MCP).** If `pdf_download=true`, for each retained paper attempt download via paper-search-mcp (`download_with_fallback`) using DOI → arXiv ID → title, with up to `pdf_max_retries` attempts. Save to `pdfs/{paper_id}.pdf`. Set `pdf_path`/`pdf_status` per §9. Bounded concurrency. Failures are non-fatal.

**Stage 9 — Summary Generation (LLM).** For each retained paper, generate an implementation-focused summary per §5-summary-rubric below, with up to `summary_max_retries` attempts. Greedy decoding.

*Summary rubric (every summary covers, in order):* problem addressed; methodology; key contributions; implementation relevance; practical usefulness. Concise (target 120–200 words). No marketing language; no claims not supported by title/abstract.

**Stage 10 — Output Assembly (deterministic).** Create the versioned output directory (§4.1). Write `01_filtered_literature.md`: run manifest, then paper records in rank order (§4.2–4.4). Ensure every `pdf_path` non-null entry has a corresponding file.

**Stage 11 — Update `_VERSION_LOG.md`.** Record the run: timestamp, output directory name, filter config hash, retained count. This marks the active Phase 1 output for downstream phases.

**Checkpointing.** The Orchestrator checkpoints after each stage (granularity `agent`/stage). On resume after a crash, completed stages are not re-run: e.g., a crash during Stage 8 resumes PDF downloads for papers not yet fetched, without re-searching or re-scoring. See Appendix D.

---

## 6. Ranking Strategy

### 6.1 Relevance Scoring Approach

`relevance_score ∈ [0.0, 1.0]` measures topical relevance of a paper to the configured `domains` (and the topic statement, if provided), assessed on the paper's **title and abstract only**.

**Anti-hallucination principle.** The LLM does **not** emit `relevance_score` directly. Asking a model to map a paper to a single number invites confabulation — the number is unanchored and unstable. Instead, the LLM emits a small set of **constrained, evidence-anchored classifications**, and `relevance_score` is **computed by code** (§6.1.4) from those classifications. A classification that cannot quote verbatim supporting text from the title or abstract is rejected and forced to `absent` (§6.1.3). The model therefore cannot score relevance on text that is not present. This mirrors the rubric-based Analyzer and evidence-anchoring decisions used in Phase 3 (see `research_gap_pipeline_implementation.md`, Core Decisions 2 and 3).

#### 6.1.1 What the LLM emits — `RelevanceAssessment`

For each paper, the LLM returns this structure (greedy / temperature-0 decoding). It contains **no numeric score** — only enums and verbatim quotes:

```
RelevanceAssessment:
  field: domain_matches      type: DomainMatch[]   one entry per configured domain (all domains, in config order)
  field: topic_alignment     type: enum            "direct" | "partial" | "none" | "no_topic_statement"
  field: topic_evidence      type: string | null   verbatim quote from title/abstract; required if topic_alignment ∈ {direct, partial}

DomainMatch:
  field: domain              type: string          echo of the configured domain
  field: match_level         type: enum            "central" | "substantial" | "peripheral" | "absent"
  field: evidence_quote      type: string | null   EXACT substring of title or abstract (≤ 200 chars); required unless match_level = absent
  field: evidence_location   type: enum | null     "title" | "abstract"; required unless match_level = absent
```

The model is instructed: classify conservatively — when genuinely uncertain between two levels, choose the **lower**. This prevents score inflation. Every non-`absent` level must be justified by an exact quote.

#### 6.1.2 Level definitions (the rubric)

The model selects each `match_level` against these fixed definitions — not a holistic impression:

```
For a configured domain D, relative to the paper's title + abstract:

central     The paper's PRIMARY subject is D (or an unambiguous synonym).
            The main contribution is in D. Typically D appears in the title
            or in the first sentence of the abstract as the object of the
            contribution (e.g. "We propose a retrieval-augmented agent ...").

substantial D is a MAJOR component of the work but not its sole focus —
            e.g. one of two methods, a core evaluation setting, or a primary
            mechanism the contribution depends on.

peripheral  D is MENTIONED, used as a tool, or appears only in motivation /
            related-work framing. The paper's contribution is not about D.

absent      Neither D nor a clear synonym appears in the title or abstract.
```

`topic_alignment` (only meaningful when a topic statement is supplied; otherwise the model returns `no_topic_statement`):

```
direct      The paper addresses the SPECIFIC research question in the topic statement.
partial     The paper addresses PART of that question, or a closely adjacent version.
none        The paper is in-domain but does NOT address the specific question.
```

#### 6.1.3 Evidence-anchoring rule (the enforcement)

This rule is what makes the score un-hallucinable, and it is enforced in **code**, not by trust:

```
For each DomainMatch with match_level ≠ absent:
  1. evidence_quote MUST be a verbatim substring of the paper's title or abstract
     (exact match after Unicode NFC normalization + whitespace collapse).
  2. evidence_location MUST correctly identify where the quote occurs.

VALIDATION (deterministic, post-LLM):
  - If evidence_quote is null, empty, or NOT found verbatim in the stated
    location → the DomainMatch is INVALID → match_level is forced to "absent"
    and the quote is discarded.
  - Same rule applies to topic_evidence for topic_alignment ∈ {direct, partial};
    a missing/invalid quote forces topic_alignment = "none".

A paper whose every DomainMatch is forced to "absent" scores 0.0 — i.e. a paper
the model cannot ground in the text is treated as not relevant, never guessed.
```

Because the validator only accepts quotes that actually occur in the source, the model cannot invent relevance. The worst case of a hallucinated quote is that the match is discarded (conservative), never that a fabricated relevance inflates the score.

#### 6.1.4 Deterministic score computation (code, not LLM)

After validation, code computes the score from the (validated) classifications. The LLM is not in this step.

```
points(match_level):  central = 1.00,  substantial = 0.70,
                      peripheral = 0.35,  absent = 0.00

best        = max over all domains of points(match_level)        # primary relevance
n_strong    = count of domains with match_level ∈ {central, substantial}
multi_bonus = 0.05 if n_strong ≥ 2 else 0.00                     # one small, capped bonus

topic_factor:  direct = 1.00,  partial = 0.90,  none = 0.80,
               no_topic_statement = 1.00                          # no penalty when no topic given

raw              = (best + multi_bonus) * topic_factor
relevance_score  = round( min(1.0, raw), 2 )                      # 2-decimal precision
```

The same validated classifications always yield the same number. There is no model variance in the arithmetic, and the enum+quote outputs the model does produce are far more stable under greedy decoding than a free float.

#### 6.1.5 Equivalence to the descriptive bands

The formula reproduces the original descriptive bands as an *output*, not as something the model picks:

| Validated classification (no topic statement) | Computed score | Band |
|---|---|---|
| `central` (best = 1.00) | 1.00 | 0.90–1.00 — directly on a configured domain |
| two `central`/`substantial` (1.00 + 0.05 bonus) | 1.00 (capped) | 0.90–1.00 |
| `substantial` (best = 0.70) | 0.70 | 0.70–0.89 — clearly relevant |
| `central` but topic `none` (1.00 × 0.80) | 0.80 | 0.70–0.89 — in-domain, not the specific question |
| `peripheral` (best = 0.35) | 0.35 | 0.30–0.49 — weak; shares broad field |
| all `absent` (best = 0.00) | 0.00 | 0.00–0.29 — not relevant |

With the default `relevance_threshold` of 0.50 (§3.2), this keeps `central` and `substantial` papers and drops `peripheral`/`absent` — a sensible cut, adjustable by the researcher.

#### 6.1.6 Optional signal blending

Off by default. When enabled (extensibility, §13), the **computed** `llm_relevance` from §6.1.4 is blended with light, normalized metadata signals that never dominate:
```
final = 0.80 * llm_relevance
      + 0.12 * citation_signal     # log-scaled, normalized to [0,1], capped
      + 0.08 * recency_signal      # linear within [years.start, years.end]; newest = 1.0
```
`citation_signal` is log-scaled and low-weighted so recent papers (few citations) are not unfairly penalized. By default (`blend=false`), `relevance_score == llm_relevance`.

**Recorded for audit.** The full `RelevanceAssessment` (validated match levels + quotes) is written to the run log for every scored paper, so any score is traceable to the exact text that produced it. (Emitting the breakdown into `01_filtered_literature.md` itself is an optional schema extension — §13, item 9.)

Conferences and years contribute **nothing** to the score — they are hard filters (binary pass/fail), not relevance signals.

### 6.2 Sorting Behavior

Retained papers are sorted by `relevance_score` **descending**. The most relevant paper is rank 1 and `paper_001`.

### 6.3 Tie-Breaking Strategy

When `relevance_score` ties (compared at 2-decimal precision), apply this fully deterministic chain in order until broken:
1. **Higher relevance_score** (full precision, before rounding).
2. **More recent `year`** (larger year first; null year sorts last).
3. **Higher `citation_count`** (null sorts as 0).
4. **Source priority**: `papersflow` > `paper_search` > `merged` is *not* used for ordering; instead use the next rule.
5. **Lexicographic by normalized title** (ascending) — guarantees a total order independent of input order.

Because rule 5 is a total order on distinct papers (titles are unique after dedup), ranking is always fully determined. `paper_id` is assigned strictly from the final rank, so identical inputs yield identical IDs.

---

## 7. Deduplication Strategy

### 7.1 Similarity Method

Deduplication uses **title similarity** as the required signal, computed on normalized titles:

- **Normalization:** lowercase; strip punctuation; collapse internal whitespace to single spaces; trim. (Stopword removal is NOT applied, to avoid collapsing distinct short titles.)
- **Similarity metric:** token-set ratio — the size of the intersection of title token sets over the size of the union (Jaccard on tokens), combined with a normalized Levenshtein ratio on the normalized strings; the similarity is the **maximum** of the two. This catches both word-order variants and minor character-level differences.
- Similarity is a float in `[0.0, 1.0]`.

### 7.2 Threshold

Two papers are duplicates if **either**:
- their DOIs are equal (case-insensitive), **or** their arXiv IDs are equal — exact-identifier match, the strongest signal; **or**
- their title similarity ≥ `options.dedup_threshold` (default **0.90**).

Exact-identifier match overrides title similarity in both directions: equal identifiers are duplicates even if titles differ (e.g., revised titles); unequal identifiers with high title similarity are still treated as duplicates (catches preprint vs camera-ready with no shared DOI).

### 7.3 Conflict Resolution Strategy

When a duplicate group is collapsed into one canonical record:
1. **Venue/year:** prefer the record with a resolvable published venue over a preprint; adopt its `conference` and `year`.
2. **Identifiers:** retain `doi` if any member has one; retain `arxiv_id` if any member has one (needed for PDF fallback even when a published DOI exists).
3. **Code link:** if any member has an explicit code link in metadata, the merged record carries it (§8); if multiple differ, choose deterministically — explicit `code`/`repository` field over a generic URL, then lexicographically smallest URL.
4. **Abstract / metadata:** keep the most complete (longest non-empty abstract; most populated fields).
5. **Authors:** take the longest author list; if equal length, the published record's.
6. **`source`:** set to `merged` when members came from different sources; otherwise the single source.
7. **`found_by_queries`:** union across members (internal, not emitted).

Deduplication is order-independent: grouping is by identifier and by a similarity graph (connected components), so the canonical record is the same regardless of input order.

### 7.4 Venue Canonicalization (supporting)

Hard conference filtering and conflict resolution both rely on resolving inconsistent venue strings ("CVPR", "IEEE/CVF Conference on Computer Vision and Pattern Recognition", a DBLP key) to a canonical key. The implementation ships a `VenueAliasMap` (Appendix B) and resolves deterministically: normalize → match against known names/aliases/canonical key → match ISSN → match DBLP key → else `null`. Unresolved venues fail an `include`-mode conference filter unless the preprint exception applies.

---

## 8. Repository Detection Logic

Repository detection is **strictly metadata-only**. No GitHub API calls, no web searches, no repository discovery of any kind beyond the metadata returned by PapersFlow and paper-search-mcp.

### 8.1 Detection Procedure

For each retained paper, inspect its aggregated metadata for an explicit code link:
1. Check explicit code/repository metadata fields provided by the sources (e.g. a `code`, `repository`, or `externalIds`/Papers-With-Code field, when present).
2. Check any URL fields in the metadata for links whose host is a known code-hosting domain: `github.com`, `gitlab.com`, `bitbucket.org`, `huggingface.co` (models/datasets/spaces), `codeberg.org`. (This set is configurable; see §13.)
3. A link qualifies only if it is **present in the metadata**. Links mentioned only inside the PDF body or abstract prose are out of scope for v2.0 (see §13 for a future option).

### 8.2 Field-Setting Rules

```
IF an explicit code link is found in metadata:
    git_exists = true
    code_url   = <the chosen link>     # selection per §7.3 rule 3 if multiple
ELSE:
    git_exists = false
    code_url   = null
```

### 8.3 Invariant

The output MUST satisfy, for every paper record:
```
git_exists == (code_url != null)
```
This invariant is validated at Stage 10. No external lookup may set `git_exists=true` without a metadata-sourced `code_url`.

---

## 9. PDF Management Workflow

### 9.1 Download Workflow

For each retained paper (in rank order, bounded concurrency), when `options.pdf_download=true`:
1. Acquire via paper-search-mcp `download_with_fallback`, supplying identifiers in priority order: **DOI → arXiv ID → normalized title**.
2. The tool falls back across its sources to obtain a PDF.
3. Retry up to `options.pdf_max_retries` times with exponential backoff on transient failures.
4. On success, persist the bytes to `pdfs/{paper_id}.pdf`.

### 9.2 Naming Convention

`pdfs/{paper_id}.pdf`, e.g. `pdfs/paper_001.pdf`. Rationale: `paper_id` is unique and deterministic (assigned by rank, §6.3), guaranteeing collision-free, directly linkable filenames. The human-readable title lives in the record's `title` field; the filename stays machine-stable. (A title-slug naming option is available via extensibility, §13.)

### 9.3 Storage Location

All PDFs live in the `pdfs/` subdirectory of the run's output directory (§4.1). No PDFs are written elsewhere.

### 9.4 Handling Failed Downloads

A failed download is **non-fatal** (NFR-5). The paper is retained with explicit markers:
```
pdf_path:   null
pdf_status: failed
```
The failure reason is recorded in the run log (not in the MD, to keep the record schema clean). The paper's metadata and summary are still produced. The funnel increments `pdfs_failed`. When `pdf_download=false`, every record has `pdf_path: null` and `pdf_status: skipped`.

### 9.5 Metadata ↔ PDF Linking

The authoritative link between an index entry and its file is the `pdf_path` field:
- `pdf_status: ok` ⇒ `pdf_path` is a relative path `pdfs/{paper_id}.pdf` that MUST exist on disk.
- `pdf_status: failed` or `skipped` ⇒ `pdf_path` is `null`.

Downstream consumers resolve a paper's PDF by joining the output directory with `pdf_path`. Because the filename derives from `paper_id`, the link is recomputable even without reading `pdf_path`.

---

## 10. Error Handling

| ID | Condition | Severity | Handling |
|---|---|---|---|
| EH-1 | Invalid filter config (bad types, `years.start > years.end`, threshold out of range) | Fatal | Fail fast at Stage 0 with a precise message; no output directory created. |
| EH-2 | One discovery source errors (PapersFlow *or* paper-search-mcp) | Recoverable | Log; continue with the other source; note degraded coverage in the run log. |
| EH-3 | Both discovery sources error | Fatal | Abort with a clear message; no partial index written. |
| EH-4 | Missing metadata fields on a candidate (no abstract, no year, etc.) | Recoverable | Use `null`; the candidate proceeds (year-null handling per §5 Stage 4). |
| EH-5 | PDF download fails after retries | Non-fatal | `pdf_path=null`, `pdf_status=failed`; increment `pdfs_failed`; keep paper. |
| EH-6 | LLM summary generation fails after retries | Non-fatal | Emit summary body: `Summary generation failed.` and set an internal `summary_status=failed` in the run log; keep paper and all metadata. |
| EH-7 | Zero candidates survive hard filtering or fall below threshold | Non-fatal (guided) | Do not crash. Write `01_filtered_literature.md` with the run manifest and a `## Notes` section reporting the funnel and recommending which filter to relax (most often `conferences`, `years`, or `relevance_threshold`). `papers_total=0`. |
| EH-8 | Output directory already exists and `overwrite=false` | Recoverable | Increment `_v{N}` until an unused directory name is found (§4.1). |
| EH-9 | Output write failure (disk/IO) | Fatal | Abort; the last checkpoint allows `anvesha resume` to retry assembly. |
| EH-10 | Repository detection encounters malformed URL in metadata | Non-fatal | Ignore the malformed link; if no valid code link remains, `git_exists=false`. Never throws. |

General principles: configuration and total-source failures are fatal; per-paper failures are isolated and surfaced as explicit field/log markers; the feature never silently drops a retained paper due to an artifact (PDF/summary) failure.

---

## 11. Example Configuration

A complete, runnable configuration for an LLM-Agents / RAG literature filter over 2023–2025:

```yaml
# Filter Literature — Phase 1 configuration
conferences:
  - NeurIPS
  - ICML
  - CVPR

domains:
  - LLM Agents
  - Retrieval-Augmented Generation

years:
  start: 2023
  end: 2025

options:
  max_papers: 40
  relevance_threshold: 0.50
  dedup_threshold: 0.90
  max_results_per_source: 100
  max_queries: 6
  include_preprints: true
  pdf_download: true
  pdf_max_retries: 3
  summary_max_retries: 2
  output_root: "."
  output_version: 1
  dirname_include_domain: true
  dirname_include_years: true
  language: "en"
```

This produces an output directory named:

```
filtered_literature_v1_llm_agents_2023_2025/
```

### Anvesha workspace form

In the Anvesha `.anvesha/config.yaml`, the same filters live under the Phase 1 block (consistent with ANVESHA.md's per-phase config convention):

```yaml
phases:
  1:
    inputs:
      - "README.md"
    output: "filtered_literature/01_filtered_literature.md"
    tools:
      - "papersflow.*"
      - "paper-search.search_papers"
      - "paper-search.download_with_fallback"
      - "filesystem.*"
    filters:
      conferences: ["NeurIPS", "ICML", "CVPR"]
      domains: ["LLM Agents", "Retrieval-Augmented Generation"]
      years: { start: 2023, end: 2025 }
    options:
      max_papers: 40
      relevance_threshold: 0.50
      pdf_download: true
```

---

## 12. Example Output

Below is an illustrative `01_filtered_literature.md` with a run manifest and two paper records — one with a metadata code link and a successful PDF download, one without a code link and a failed download. (Values are illustrative.)

```markdown
---
document_type: filtered_literature_index
schema_version: "2.0"
generated_at: 2026-01-15T09:42:11Z
generator: anvesha.phase01_filter
filters:
  conferences:
    - NeurIPS
    - ICML
    - CVPR
  domains:
    - LLM Agents
    - Retrieval-Augmented Generation
  years:
    start: 2023
    end: 2025
counts:
  identified: 412
  after_deduplication: 318
  after_hard_filter: 96
  screened: 96
  retained: 40
  pdfs_downloaded: 38
  pdfs_failed: 2
  with_code: 23
papers_total: 40
output_dir: filtered_literature_v1_llm_agents_2023_2025
---

---
paper_id: paper_001
title: Toolformer-Style Self-Supervised Tool Use for Retrieval-Augmented Agents
authors:
  - A. Researcher
  - B. Scientist
conference: NeurIPS
year: 2024
domain:
  - LLM Agents
  - Retrieval-Augmented Generation
paper_url: https://example.org/abs/2024.00001
pdf_path: pdfs/paper_001.pdf
pdf_status: ok
code_url: https://github.com/example/toolformer-rag
git_exists: true
relevance_score: 0.94
rank: 1
doi: 10.0000/neurips.2024.00001
arxiv_id: 2401.00001
source: merged
---

## LLM Summary

Problem: retrieval-augmented LLM agents struggle to learn when and how to
invoke external tools without large amounts of supervised tool-use data.
Methodology: the paper introduces a self-supervised objective that lets the
model annotate its own training corpus with tool calls, filtering them by
whether the call reduces next-token loss. Key contributions: a data-efficient
self-supervision scheme, a retrieval-aware tool API, and benchmarks on
multi-hop QA. Implementation relevance: the training loop is reproducible from
the described objective and the released repository; the tool API is a thin
wrapper compatible with standard retrieval backends. Practical usefulness:
directly applicable to building RAG agents that learn tool use from unlabeled
corpora, with a clear path to integration via the provided code.

---
paper_id: paper_002
title: Memory Compression Strategies for Long-Horizon LLM Agents
authors:
  - C. Author
conference: ICML
year: 2025
domain:
  - LLM Agents
paper_url: https://example.org/abs/2025.00042
pdf_path: null
pdf_status: failed
code_url: null
git_exists: false
relevance_score: 0.88
rank: 2
doi: null
arxiv_id: 2505.00042
source: paper_search
---

## LLM Summary

Problem: long-horizon agents accumulate context that exceeds the model's
window, degrading performance and inflating cost. Methodology: the paper
compares three memory-compression strategies — recursive summarization,
salience-based eviction, and learned key-value compression — under a shared
agent harness. Key contributions: a controlled comparison, a salience metric
for eviction, and cost/quality trade-off curves. Implementation relevance:
the strategies are described at the algorithm level and can be reimplemented,
though no repository is linked in the metadata, so reproduction relies on the
paper's pseudocode. Practical usefulness: useful for teams choosing a memory
strategy for production agents; the trade-off curves inform the choice, but
implementation effort is higher absent released code.
```

---

## 13. Future Extensibility Considerations

These are explicitly **out of scope for v2.0** but the design accommodates them:

1. **Additional discovery sources.** The search layer is source-agnostic behind a normalizer. Adding Semantic Scholar / OpenAlex / Zotero MCPs requires only a new normalizer mapping and a `sources` entry — no change to filtering, ranking, or output.
2. **In-PDF code-link extraction.** A future `repo_detection.scan_pdf=true` option could parse the downloaded PDF for code URLs. This would relax the strict metadata-only rule and so is gated behind an explicit opt-in; the default remains metadata-only (§8).
3. **Configurable code-host domains.** The known-host list in §8.1 can be exposed as `repo_detection.hosts` for niche forges.
4. **Title-slug PDF naming.** A `pdf_naming: "id" | "id_slug"` option could produce `paper_001_toolformer_style.pdf` while keeping `paper_id` linkage. Default stays `id` for strict determinism (§9.2).
5. **Signal blending for ranking.** The blended scoring in §6.1 is built but off by default; a `ranking.blend=true` switch plus weight overrides exposes it.
6. **Multi-domain directory slugs.** `dirname_include_domain` currently uses the first domain; a `dirname_domain_mode: "first" | "all"` option could join all domain slugs.
7. **Per-conference query specialization.** The Query Builder could tailor queries per target venue; deferred to keep query generation deterministic and simple.
8. **Incremental updates.** A future mode could diff against a prior versioned directory and fetch only new papers, appending to the index — useful for periodic re-runs of a standing filter.
9. **Emit the relevance breakdown.** The validated `RelevanceAssessment` (per-domain match levels and evidence quotes, §6.1.1) is recorded in the run log today. A future schema version could emit it into each paper record — giving downstream consumers the exact text that justified the score, fully traceable.

---

## Appendix A — Architecture Diagram

```
[conferences + domains + years]  [topic statement (optional)]
                    │
              ┌─────▼─────┐
              │Orchestrator│  (stateful controller; funnel counts; checkpoints)
              └─────┬─────┘
   Stage 1  ┌───────▼────────┐
   (LLM)    │ Query Builder  │  domains → diverse search queries
            └───────┬────────┘
   Stage 2  ┌───────▼────────┐   PapersFlow + paper-search-mcp (concurrent)
   (MCP)    │ Multi-Source   │ ─────────────────────────────────► [identified]
            │   Search       │
            └───────┬────────┘
   Stage 3  ┌───────▼────────┐
            │ Normalize +    │   DOI/arXiv exact → title-similarity dedup
            │ Deduplicate    │ ─────────────────────────────────► [after_deduplication]
            └───────┬────────┘
   Stage 4  ┌───────▼────────┐
            │ Hard Filter    │   conference (canonicalized) → year → language
            │ Gate           │ ─────────────────────────────────► [after_hard_filter]
            └───────┬────────┘
   Stage 5  ┌───────▼────────┐
   (LLM)    │ Relevance      │   score vs domains on title+abstract
            │ Scoring        │ ─────────────────────────────────► [screened]
            └───────┬────────┘
   Stage 6  ┌───────▼────────┐
            │ Rank & Select  │   sort + tie-break; cap max_papers; assign paper_id
            └───────┬────────┘ ─────────────────────────────────► [retained]
   Stage 7  ┌───────▼────────┐
            │ Repo Detection │   metadata-only → git_exists, code_url
            └───────┬────────┘
   Stage 8  ┌───────▼────────┐   paper-search-mcp download_with_fallback
   (MCP)    │ PDF Acquisition│ ─────────────────────────────────► pdfs/paper_NNN.pdf
            └───────┬────────┘
   Stage 9  ┌───────▼────────┐
   (LLM)    │ Summaries      │   implementation-focused, per paper
            └───────┬────────┘
   Stage 10 ┌───────▼────────┐
            │ Output Assembly│   versioned dir + 01_filtered_literature.md
            └───────┬────────┘
                    ▼
   filtered_literature_v1_llm_agents_2023_2025/
   ├── 01_filtered_literature.md   →  INPUT to Phase 2 (Literature Survey)
   └── pdfs/
```

---

## Appendix B — Internal Type Definitions

These types are internal to the implementation (not all are emitted; the emitted schema is §4.3).

**PaperCandidate** (post-normalization)
```
field: candidate_id      type: string     internal, pre-rank
field: title             type: string
field: authors           type: string[]
field: abstract          type: string | null
field: venue_raw         type: string | null
field: venue_canonical   type: string | null
field: year              type: integer | null
field: citation_count    type: integer | null
field: doi               type: string | null
field: arxiv_id          type: string | null
field: paper_url         type: string | null
field: metadata_code_url type: string | null   raw code link found in metadata (§8)
field: language          type: string | null
field: is_preprint       type: boolean
field: source            type: enum: "papersflow" | "paper_search"
field: found_by_queries  type: string[]
```

**ScoredCandidate** (post-Stage 5)
```
field: candidate_id      type: string
field: assessment        type: RelevanceAssessment   validated per-domain matches + topic alignment (§6.1.1)
field: relevance_score   type: float   [0.0, 1.0]     computed from assessment (§6.1.4)
field: matched_domains   type: string[]               domains with match_level ≠ absent (post-validation)
```

**RelevanceAssessment / DomainMatch** — the LLM's anchored classification output and the basis for the computed score. Full schema and the deterministic computation are defined in §6.1.1–§6.1.4. `DomainMatch.evidence_quote` must be a verbatim substring of the title or abstract; invalid quotes force `match_level = absent` (§6.1.3).

**RetainedPaper** (post-Stage 6; basis for the emitted record)
```
field: paper_id          type: string   "paper_{NNN}"
field: rank              type: integer
field: ...               (all emitted fields per §4.3, populated through Stages 7–9)
field: pdf_status        type: enum: "ok" | "failed" | "skipped"
field: summary           type: string
field: summary_status    type: enum: "ok" | "failed"   (internal/log only)
```

**VenueAlias** (venue canonicalization)
```
field: canonical_key     type: string     e.g. "NeurIPS"
field: full_names        type: string[]   e.g. "Advances in Neural Information Processing Systems"
field: aliases           type: string[]   e.g. "NIPS"
field: dblp_keys         type: string[]
field: issns             type: string[]
field: venue_type        type: enum: "conference" | "journal" | "workshop"
```

**FilterCounts** (the funnel; emitted as `counts` in the manifest)
```
identified, after_deduplication, after_hard_filter, screened, retained,
pdfs_downloaded, pdfs_failed, with_code   — all integers
```

---

## Appendix C — SDK Package Structure

Phase 1 lives under `anvesha/phases/phase01_filter/` and uses shared `core/` infrastructure identically to every phase (ANVESHA.md §7). It imports nothing from other phase modules.

```
phase01_filter/
├── __init__.py
├── pipeline.py                 # FilterLiteraturePipeline — Phase 1 entry point
│                               # accepts ResearchProject; runs Stages 0–11;
│                               # writes the versioned output directory
├── orchestrator.py             # stage sequencing, FilterCounts, checkpoint hooks
├── state.py                    # PhaseState (intermediate stage outputs)
│
├── schemas/
│   ├── config.py               # FilterConfig (conferences/domains/years) + Options
│   ├── candidate.py            # PaperCandidate, ScoredCandidate, RetainedPaper
│   ├── counts.py               # FilterCounts
│   └── record.py               # emitted record + run-manifest serializers (§4.3 order)
│
├── stages/
│   ├── query_builder.py        # Stage 1 (LLM)
│   ├── search.py               # Stage 2 (PapersFlow + paper-search-mcp, concurrent)
│   ├── dedup.py                # Stage 3 (identifier + title similarity)
│   ├── hard_filter.py          # Stage 4 (conference/year/language)
│   ├── relevance.py            # Stage 5 (LLM, batched)
│   ├── rank.py                 # Stage 6 (deterministic sort + tie-break)
│   ├── repo_detect.py          # Stage 7 (metadata-only; §8)
│   ├── pdf_fetch.py            # Stage 8 (download_with_fallback; §9)
│   ├── summarize.py            # Stage 9 (LLM)
│   └── assemble.py             # Stage 10 (versioned dir + markdown writer)
│
└── venues/
    ├── alias_map.py            # built-in VenueAliasMap
    └── canonicalize.py         # deterministic venue resolution (§7.4)
```

Boundary compliance (ANVESHA.md §12): no cross-phase imports; all file I/O via `workspace/phase_io.py`; LLM and MCP access via injected clients; entry point takes a `ResearchProject`.

---

## Appendix D — Shared Infrastructure (reference)

Phase 1 reuses Anvesha's shared infrastructure rather than redefining it. See `ANVESHA.md`:

- **LLM adapters and backend selection** — §7.2, §10. Phase 1 is light on compute: Query Builder, Relevance Scoring, and Summaries are short calls; a mid-tier local model at low/medium reasoning is sufficient.
- **Checkpoint and resume** — §4.4. Phase 1 checkpoints per stage; resume continues mid-PDF-download or mid-summarization without re-searching or re-scoring.
- **Configuration precedence and the `Phase` base class** — §7, §8. The filter block and options are injected via the loaded config; no stage reads config files directly.
- **Boundary rules** — §12 (identical hard constraints for all phases).
- **Versioning and `_VERSION_LOG.md`** — §11. Phase 1 uses directory-level versioning (`_v{N}`) because its output is a directory (index + PDFs); the version log records the active directory for downstream phases.

---

*End of specification. Phase 1 (Filter Literature) v2.0.*
