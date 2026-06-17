# Literature Survey — Technical Implementation Specification
## Phase 2 of the Anvesha Pipeline · v1.0

> **Audience:** (1) engineers implementing the feature, and (2) downstream LLM workflows (Phase 3, Research Gaps) that consume its output.
>
> **Scope of this document:** A complete, prescriptive implementation specification for the **Literature Survey** feature (Phase 2 of Anvesha). It defines objectives, scope, inputs/outputs, review strategy and methodology, the step-by-step processing workflow, the per-paper data-extraction template, dataset/benchmark and architecture extraction, the cross-paper synthesis strategy, the quality-assessment approach, the output document template, and the evaluation metrics that make the review systematic and reproducible.
>
> **Where Phase 2 sits:** Phase 1 (Filter Literature) produced a curated reading list `01_filtered_literature.md` plus downloaded PDFs in `pdfs/`. Phase 2 **deep-reads those papers** and synthesizes them into a single document, `02_literature_survey.md`, which is the sole input to Phase 3 (Research Gaps). Phase 1 worked on titles/abstracts (shallow, high-volume); Phase 2 works on full text (deep, lower-volume).
>
> **Relationship to Anvesha:** See `ANVESHA.md` for the eleven-phase architecture, the `Phase` base class, configuration system, and shared infrastructure (LLM adapters, checkpointing, boundary rules). This spec references that material rather than repeating it.
>
> **Carried-forward commitment — evidence anchoring.** Every extracted claim about a paper MUST be anchored to that paper's own text (a verbatim quote plus a locator), validated in code. A claim that cannot be anchored is dropped, never fabricated. This is the same anti-hallucination mechanism specified for Phase 1 scoring (`filter_literature_implementation.md` §6.1.3) and the Phase 3 evidence-anchoring decision.

---

## Table of Contents

1. Objectives
2. Scope
3. Inputs
4. Output and Downstream Contract
5. Survey Strategy and Methodology
6. Processing Workflow (step-by-step)
7. Per-Paper Data-Extraction Template
8. Datasets and Benchmarks (extraction + landscape)
9. Frameworks and Network/Architecture (extraction + analysis)
10. Cross-Paper Synthesis Strategy
11. Quality-Assessment Approach
12. Output Document Template
13. Evaluation Metrics
14. Error Handling
15. Example Output (excerpt)
16. Future Extensibility

Appendix A — Architecture Diagram
Appendix B — Internal Type Definitions
Appendix C — SDK Package Structure
Appendix D — Shared Infrastructure (reference)

---

## 1. Objectives

Phase 2 produces a **systematic, reproducible, academically rigorous literature survey** of the papers retained by Phase 1. Concretely it must:

**O-1 — Comprehensive coverage.** Produce a structured summary of **every** retained paper, extracted from full text where a PDF is available.

**O-2 — Structured, anchored extraction.** Extract a consistent, evidence-anchored data record per paper covering problem, method, architecture/network, **datasets used**, benchmarks/metrics, frameworks/tools, results, contributions, limitations, and stated future work.

**O-3 — Cross-paper synthesis.** Synthesize across the corpus: a thematic taxonomy of approaches, a datasets-and-benchmarks landscape, an architecture/framework analysis, a comparative analysis on shared benchmarks, chronological trends, consensus findings, contradictions, and under-explored directions.

**O-4 — Quality assessment.** Assess each paper's methodological quality against a fixed rubric, and characterize the reliability of the corpus as a whole.

**O-5 — Gap-ready output.** Emit a single document, `02_literature_survey.md`, deliberately structured so Phase 3 (Research Gaps) can mine it: per-paper limitations and future work, cross-paper contradictions, dataset/benchmark coverage gaps, and explicitly flagged under-explored directions.

**O-6 — Reproducibility.** Document the methodology, record provenance (which Phase 1 run, which papers), and make extraction deterministic and auditable so the survey can be regenerated and verified.

---

## 2. Scope

### 2.1 In Scope
- Reading and extracting from the PDFs (and abstracts) of papers in the Phase 1 output directory.
- Structured per-paper extraction and rubric-based quality assessment.
- Cross-paper thematic, dataset/benchmark, architectural, comparative, and trend synthesis.
- Production of the single survey document and its provenance manifest.

### 2.2 Out of Scope
- **Discovery / retrieval.** Phase 2 does not search for papers or download PDFs — that is Phase 1. Phase 2 reads only what Phase 1 produced. (If a paper's PDF failed to download in Phase 1, Phase 2 extracts from its abstract and flags the reduced depth — §6, Stage 1; it does not attempt re-download.)
- **Gap identification.** Phase 2 surfaces the *raw material* for gaps (limitations, contradictions, coverage holes) but does not itself decide what the research gaps are — that is Phase 3.
- **Repository discovery.** Code availability is carried over from Phase 1 metadata (`git_exists`, `code_url`); Phase 2 performs no repository search.
- **Citation-graph expansion.** Phase 2 surveys the fixed Phase 1 corpus; it does not pull in new references. (Citation-graph expansion is an extensibility item, §16.)

### 2.3 Corpus Boundary
The survey covers exactly the **retained** papers in the active Phase 1 output directory — no more, no fewer. If the researcher wants a different corpus, they adjust the Phase 1 filters and re-run Phase 1, then re-run Phase 2.

---

## 3. Inputs

Phase 2 reads the active Phase 1 output directory (its path is recorded in `_VERSION_LOG.md`; default `filtered_literature[_v{N}_...]/`).

| Input | Source | Use |
|---|---|---|
| `01_filtered_literature.md` | Phase 1 | The paper index: per-paper metadata (paper_id, title, authors, venue, year, doi, arxiv_id, paper_url, `pdf_path`, `pdf_status`, `code_url`, `git_exists`, `relevance_score`, `rank`, `domain`) parsed via the Phase 1 grammar (`filter_literature_implementation.md` §4.4). |
| `pdfs/paper_{NNN}.pdf` | Phase 1 | Full text for extraction. Linked from each record's `pdf_path`. |
| Topic statement (optional) | `README.md` | Sharpens synthesis framing and relevance of the narrative; never overrides extracted facts. |

### 3.1 Options

```yaml
phases:
  2:
    inputs:
      - "filtered_literature/01_filtered_literature.md"   # resolved to active Phase 1 dir
    output: "phases/02_literature_survey.md"
    tools:
      - "filesystem.*"                 # read PDFs + index, write the survey
    options:
      pdf_text_extractor: "pymupdf"    # primary PDF text extractor
      ocr_fallback: true               # OCR scanned PDFs that yield no text layer
      section_segmentation: true       # segment full text into IMRaD-style sections
      extraction_batch_size: 1         # papers per extraction LLM call (1 = max fidelity)
      synthesis_context_mode: "records"# synthesize from extraction records, not raw PDFs
      quality_assessment: true         # run the per-paper quality rubric (§11)
      min_anchor_coverage: 0.80        # min fraction of extracted claims that must be anchored
      verification_pass: true          # run the synthesis self-consistency check (§6 Stage 6)
      taxonomy_max_themes: 8           # cap on top-level thematic clusters
```

---

## 4. Output and Downstream Contract

### 4.1 Single Artifact
Phase 2 produces exactly one file: **`02_literature_survey.md`**, written to the workspace `phases/` directory. It is the **sole input** to Phase 3 (Research Gaps), which reads it as the "Literature MD."

### 4.2 Dual Nature: narrative + machine-readable
The document is simultaneously:
- a **human-readable narrative survey** (objectives, methodology, synthesis sections), and
- a **machine-consumable structure** for Phase 3: a run manifest (YAML), per-paper structured records in a fixed format, and synthesis sections in which **every claim cites the paper(s) by `paper_id`**.

The full template is §12.

### 4.3 Downstream Contract (what Phase 3 relies on)
The Research Gaps Gap Finder (see `research_gap_pipeline_implementation.md` §5.2) anchors gap evidence in this document. Phase 2 therefore guarantees:
1. **Per-paper traceability** — each paper has a structured record keyed by `paper_id`, so a gap can be attributed to specific papers.
2. **Quotable specificity** — synthesis statements are concrete and attributed (`[paper_003]`), so the Gap Finder can cite verbatim text with `anchor_source: "literature_md"`.
3. **Gap-feeding sections** present and clearly delimited: per-paper **stated limitations** and **stated future work** (open problems), a **Contradictions & Open Debates** section (cross-paper conflicts), a **Datasets & Benchmarks** section with **coverage gaps** (missing-benchmark signal), and an **Under-Explored Directions** section (unexplored-direction signal). These map one-to-one to the Discovery-Mode gap types in Phase 3.

### 4.4 Versioning
Re-running Phase 2 auto-versions the output (`02_literature_survey_v2.md`, …) per ANVESHA.md §11; `_VERSION_LOG.md` records the active version. Looping back from a later phase that invalidates the survey triggers a new version.

---

## 5. Survey Strategy and Methodology

The methodology adapts established systematic-review practice (structured extraction, quality appraisal, thematic + comparative synthesis) to a technical ML/CV/LLM corpus. It is the documented, reproducible procedure required by O-6.

### 5.1 Methodological Principles (non-functional requirements)

**M-1 — Evidence anchoring (hard requirement).** Every extracted field carries an `evidence_locator` (verbatim quote + section/page). Quotes are validated against the extracted PDF text; an unvalidated quote forces the field to be dropped or marked `unverified`, never fabricated (§7.3). The corpus-level **anchoring rate** is an output metric (§13).

**M-2 — Depth honesty.** A paper extracted from full text is marked `extraction_depth: full_text`; a paper whose PDF was unavailable is marked `abstract_only`. Synthesis and quality assessment explicitly account for depth; abstract-only papers never receive fabricated full-text detail.

**M-3 — Determinism.** Extraction and quality LLM calls use greedy / temperature-0 decoding. Aggregations (dataset tallies, taxonomy assignment thresholds, comparative tables) are computed by code from the extraction records. Given the same PDFs and model, the survey is reproducible.

**M-4 — Synthesis from records, not raw text.** Cross-paper synthesis operates over the **validated extraction records** (`synthesis_context_mode: records`), not the raw PDFs. This bounds context, keeps synthesis grounded in already-anchored facts, and prevents re-introduction of un-anchored claims.

**M-5 — Attribution.** Every synthesis claim names its supporting paper(s) by `paper_id`. Unattributed generalizations are not permitted (validated in Stage 6).

**M-6 — Boundary compliance.** No imports from other phase modules; all file I/O via the workspace module; LLM access via the injected client; the entry point accepts a `ResearchProject` (ANVESHA.md §12).

### 5.2 Review Strategy (the approach)

1. **Normalize the corpus** — parse the index, pair each paper with its PDF, establish extraction depth.
2. **Extract uniformly** — apply one fixed extraction template (§7) to every paper, so papers are directly comparable.
3. **Appraise quality** — score each paper against one fixed rubric (§11), so the synthesis can weight reliable work.
4. **Synthesize structurally** — build a taxonomy, a datasets/benchmarks landscape, an architecture analysis, comparative tables, trends, consensus, contradictions, and under-explored directions (§10) — all from the extraction records.
5. **Verify** — check that the synthesis is consistent with and traceable to the records (Stage 6).
6. **Assemble** — emit the single survey document against a fixed template (§12).

### 5.3 Compute Profile
Phase 2 is **moderate-to-heavy**: per-paper full-text extraction is parallelizable across papers; the synthesis stage is a single large-context operation over all records and benefits from a strong model with a large context window (e.g. Nemotron 3 Super, ~1M context — see `research_gap_pipeline_implementation.md` §15.2). See Appendix D.

---

## 6. Processing Workflow (step-by-step)

The pipeline is **largely linear** with one optional verification pass. The Orchestrator drives stages and maintains progress metrics; it checkpoints per stage and (within Stage 2/3) per paper, so a crash resumes without re-extracting completed papers.

```
Stage 0  Load & pair inputs            (deterministic)
Stage 1  PDF text extraction           (deterministic; OCR fallback)
Stage 2  Per-paper structured extraction (LLM, evidence-anchored)
Stage 3  Per-paper quality assessment  (LLM, rubric-based)
Stage 4  Anchor validation             (deterministic)
Stage 5  Cross-paper synthesis         (LLM, over records)
Stage 6  Synthesis verification        (LLM + deterministic checks; optional)
Stage 7  Survey assembly               (deterministic + prose)
Stage 8  Metrics + version log         (deterministic)
```

**Stage 0 — Load & pair inputs.** Parse `01_filtered_literature.md` (Phase 1 grammar). For each paper record, resolve `pdf_path` to a file in `pdfs/`. Establish `extraction_depth`: `full_text` if the PDF exists and is readable; `abstract_only` if `pdf_status != ok` or the file is missing. Carry over `git_exists`/`code_url`. Build the working set in rank order.

**Stage 1 — PDF text extraction (deterministic).** For each `full_text` paper, extract text with `pdf_text_extractor` (default PyMuPDF). If the PDF has no text layer (scanned) and `ocr_fallback=true`, OCR it. If `section_segmentation=true`, segment into IMRaD-style sections (Abstract, Introduction, Related Work, Method, Experiments/Results, Conclusion) by heading detection; store page offsets for locators. If extraction yields no usable text, demote the paper to `abstract_only` (EH-2). `abstract_only` papers use the abstract text from the index.

**Stage 2 — Per-paper structured extraction (LLM, anchored).** For each paper, run the extraction template (§7) over its available text. The LLM returns the `ExtractionRecord` with an `evidence_locator` on every field. Greedy decoding. Batched per `extraction_batch_size` (default 1 for fidelity). For `abstract_only` papers, only fields supportable by the abstract are populated; the rest are `null` with `extraction_depth` noted — no fabrication.

**Stage 3 — Per-paper quality assessment (LLM, rubric).** For each paper, run the quality rubric (§11), producing anchored sub-scores and a computed quality grade. `abstract_only` papers receive a partial assessment flagged accordingly.

**Stage 4 — Anchor validation (deterministic).** For every extracted field and quality sub-score with an `evidence_locator`, verify the quote is a verbatim substring of the paper's extracted text (NFC + whitespace normalization). Invalid → mark the field `unverified` and exclude its quote (§7.3). Compute the per-paper and corpus **anchoring rate**; if a paper falls below `min_anchor_coverage`, flag it `low_confidence_extraction` (still included, surfaced in metrics).

**Stage 5 — Cross-paper synthesis (LLM, over records).** Operating over the validated records (not raw PDFs), produce: the thematic taxonomy (§10.1), datasets/benchmarks landscape (§8), architecture/framework analysis (§9), comparative synthesis (§10.2), chronological trends (§10.3), consensus findings (§10.4), contradictions (§10.5), and under-explored directions (§10.6). Every synthesis claim cites `paper_id`(s).

**Stage 6 — Synthesis verification (optional, `verification_pass=true`).** A self-consistency gate (a check, not an iterative loop):
- *Deterministic:* every `[paper_id]` cited in synthesis exists in the corpus; every paper appears in ≥1 synthesis section or the taxonomy; comparative-table numbers match the source records.
- *LLM:* sample synthesis claims and verify each is supported by the cited records' anchored content; flag unsupported claims for removal or qualification.
This raises synthesis fidelity without the Generator/Critic/Judge machinery reserved for Phase 3.

**Stage 7 — Survey assembly (deterministic + prose).** Assemble `02_literature_survey.md` against the §12 template: manifest, objectives/scope/methodology, corpus overview, taxonomy, per-paper records, datasets/benchmarks, architecture analysis, comparative synthesis, trends, consensus, contradictions, under-explored directions, quality summary, and the gap-analysis handoff.

**Stage 8 — Metrics + version log.** Compute the evaluation metrics (§13), embed the relevant ones in the manifest, and update `_VERSION_LOG.md`.

---

## 7. Per-Paper Data-Extraction Template

This is the fixed extraction form applied to every paper (the systematic-review "data extraction sheet"). Uniformity makes papers comparable and the synthesis aggregable.

### 7.1 `ExtractionRecord` schema

```
ExtractionRecord:
  paper_id            string         from Phase 1
  citation            string         "Authors (Year). Title. Venue."
  extraction_depth    enum           "full_text" | "abstract_only"
  problem             Field          the problem/research question addressed
  task                Field          task type (e.g., open-domain QA, agentic planning)
  method              Field          core method / key idea (2–4 sentences)
  architecture        ArchBlock      network/architecture details (§9)
  datasets_used       DatasetUse[]   datasets and their roles (§8)
  benchmarks          BenchResult[]  benchmark results reported (§8)
  metrics             Field[]        evaluation metrics used
  frameworks_tools    Field[]        frameworks/libraries (PyTorch, HF, LangChain, …)
  training_setup      Field          notable compute/hyperparameters (optional)
  key_results         Field[]        principal quantitative/qualitative findings
  contributions       Field[]        claimed contributions
  stated_limitations  Field[]        limitations the authors state           ← feeds Phase 3
  stated_future_work  Field[]        future work the authors propose          ← feeds Phase 3
  relations           Relation[]     links to other corpus papers (§10)
  quality             QualityResult  per-paper quality assessment (§11)
  flags               string[]       e.g. "abstract_only", "low_confidence_extraction"

Field:
  value               string         the extracted content
  evidence_locator    Locator | null required for full_text claims (§7.3)
  status              enum           "verified" | "unverified" | "not_stated"

Locator:
  section             string | null  e.g. "Method", "Experiments"
  page                integer | null
  quote               string         verbatim span (≤ 240 chars) from the paper text

Relation:
  related_paper_id    string         another corpus paper_id
  relation_type       enum           "builds_on" | "extends" | "contradicts" | "compares_to" | "uses_method_of"
  evidence_locator    Locator | null
```

### 7.2 Extraction guidance (per field)
- **problem / task / method** — state plainly; do not editorialize. `method.value` is the core idea, not a marketing summary.
- **contributions / key_results** — prefer the authors' own claims with a quote; mark `status: not_stated` if the paper does not state them explicitly.
- **stated_limitations / stated_future_work** — extract verbatim-grounded statements only. If the paper states none, emit an empty list (not invented limitations). These two fields are high-value for Phase 3, so accuracy matters more than completeness.
- **relations** — only assert a relation to another corpus paper when the text supports it (citation + claim); otherwise omit. `contradicts` requires a quote showing the conflicting claim.

### 7.3 Anchoring and validation (anti-hallucination)
Identical in spirit to Phase 1 §6.1.3, adapted to full text:
```
For every Field/Relation with a non-null evidence_locator (required for full_text):
  - quote MUST be a verbatim substring of the paper's extracted text
    (NFC + whitespace-collapsed comparison).
VALIDATION (Stage 4, deterministic):
  - quote missing or not found  → status = "unverified", quote discarded.
  - field genuinely absent in source → status = "not_stated", value = null.
A field can be "verified", "unverified", or "not_stated" — never a fabricated fact
presented as grounded. abstract_only papers anchor against the abstract text.
```

---

## 8. Datasets and Benchmarks (extraction + landscape)

A first-class dimension of the survey. ML/CV/LLM progress is legible largely through *which datasets and benchmarks* papers use; aggregating them reveals standards, emerging resources, and — critically for Phase 3 — **evaluation coverage gaps**.

### 8.1 Per-paper extraction

```
DatasetUse:
  name              string         canonical dataset/benchmark name (alias-normalized, §8.3)
  role              enum           "train" | "eval" | "both" | "pretrain"
  task              string         task the dataset serves (e.g., multi-hop QA)
  modality          string         text | image | video | multimodal | tabular | …
  size              string | null  as stated (e.g., "1.2M QA pairs")
  evidence_locator  Locator | null

BenchResult:
  benchmark         string         benchmark name (alias-normalized)
  metric            string         metric reported (e.g., EM, F1, mAP, accuracy)
  reported_value    string         the value the paper reports for its method
  baseline_delta    string | null  improvement vs the paper's stated baseline, if given
  evidence_locator  Locator | null
```

### 8.2 Corpus-level landscape (synthesis)
Computed by code from the per-paper `DatasetUse` / `BenchResult` records, with LLM narrative:
- **Dataset/benchmark frequency table** — dataset → list of `paper_id`s using it → task → modality → train/eval roles. Sorted by usage count.
- **Dominant vs emerging** — datasets used by many papers (de-facto standards) vs those used by one or two (emerging/niche).
- **Metric conventions per task** — which metrics are standard for each task in the corpus.
- **Coverage gaps (feeds Phase 3)** — tasks or settings that appear in the corpus's *problems* but lack a shared evaluation benchmark; benchmarks used by only a single paper (weak comparability); modalities or regimes (e.g., low-resource, long-context) with thin evaluation. Each gap is stated with the `paper_id`s that motivate it.

### 8.3 Name normalization
A `DatasetAliasMap` (Appendix B) canonicalizes dataset/benchmark names ("MS MARCO", "MSMARCO"; "ImageNet-1k", "ImageNet") so aggregation is correct. Unresolved names are kept verbatim and flagged for the researcher to alias.

---

## 9. Frameworks and Network/Architecture (extraction + analysis)

The second emphasized dimension: the architectural and framework details that determine how methods work and whether they are buildable.

### 9.1 Per-paper extraction

```
ArchBlock:
  family            string         architecture family
                                   (e.g., transformer, CNN, diffusion, retrieval-augmented,
                                    agentic-loop, mixture-of-experts, graph-neural)
  components        string[]       key building blocks (e.g., "cross-attention retriever",
                                   "tool-calling planner", "KV-cache compressor")
  novel_components  string[]       components the paper claims as new
  base_model        string | null  backbone if applicable (e.g., "LLaMA-2-7B", "ViT-L/16")
  evidence_locator  Locator | null
```
`frameworks_tools` (in the ExtractionRecord) captures libraries/frameworks explicitly named (PyTorch, JAX, HF Transformers, LangChain, vLLM, …), each anchored. `training_setup` optionally captures notable compute/hyperparameters when stated.

### 9.2 Corpus-level analysis (synthesis)
- **Architecture taxonomy** — group papers by `family`; within families, cluster by shared `components`.
- **Common building blocks** — components recurring across papers (the field's standard machinery) vs one-off novel components.
- **Backbone trends** — which base models dominate, and how that shifts across the year range.
- **Framework adoption** — which frameworks/tools the corpus relies on (and, combined with Phase 1 `git_exists`, which lines of work are most reproducible).

---

## 10. Cross-Paper Synthesis Strategy

All synthesis operates over the validated extraction records (M-4) and attributes every claim to `paper_id`(s) (M-5).

### 10.1 Thematic Taxonomy
Cluster papers into ≤ `taxonomy_max_themes` top-level themes (approach families / sub-problems), each with a short definition and the member `paper_id`s. A paper may belong to multiple themes. The taxonomy is the survey's organizing spine and reveals dense vs sparse areas of the field.

### 10.2 Comparative Synthesis
For papers sharing a benchmark (from §8), build a comparison: method → reported metric values → datasets → key differences. Compare only like-for-like (same benchmark + metric); never compare across incompatible settings. Surface trade-offs (accuracy vs cost, generality vs specialization).

### 10.3 Chronological Evolution & Trends
Order key developments across `[years.start, years.end]`: how methods, backbones, and benchmarks shifted; what was superseded; what direction the field is moving.

### 10.4 Consensus Findings
Claims the corpus broadly agrees on (supported by multiple papers), each with its supporting `paper_id`s. Consensus weighted by per-paper quality (§11) — agreement among higher-quality papers is stronger.

### 10.5 Contradictions & Open Debates  *(feeds Phase 3)*
Points where papers disagree — conflicting results on the same benchmark, opposing design claims, irreconcilable conclusions. Each contradiction names the conflicting `paper_id`s and quotes the conflicting claims. This section directly supplies Phase 3's `contradiction_between_papers` gap type.

### 10.6 Under-Explored Directions & Open Problems  *(feeds Phase 3)*
Aggregated from per-paper `stated_future_work` and `stated_limitations`, plus coverage gaps (§8.2) and sparse taxonomy regions (§10.1). Each entry is a concrete direction with the motivating `paper_id`s. This section supplies Phase 3's `open_problem_stated`, `unexplored_direction`, and `missing_benchmark` gap types.

---

## 11. Quality-Assessment Approach

A rubric-based appraisal of each paper's methodological quality — anchored, computed (not a holistic guess), and used to weight synthesis. Same anti-hallucination pattern as §7.3 and Phase 1 §6.1.

### 11.1 What the LLM emits (per paper) — anchored sub-judgments, no score

For each dimension, the LLM returns an ordinal level + an `evidence_locator`. It does **not** emit the final grade.

```
QualityAssessment:
  reproducibility   Level   # artifacts + method specification
  eval_rigor        Level   # baselines, datasets, metrics, ablations
  evidence_strength Level   # are claims supported by the reported results
  novelty           Level   # contribution distinctiveness within the corpus
  clarity           Level   # completeness/specificity of the method description

Level:
  level             enum    "strong" | "adequate" | "weak" | "not_assessable"
  evidence_locator  Locator | null   required unless "not_assessable"
```

Dimension definitions (fixed):
```
reproducibility   strong  = code AND data available (git_exists from Phase 1) + method fully specified
                  adequate= method fully specified, partial artifacts
                  weak    = key details missing; not reconstructable
eval_rigor        strong  = appropriate baselines + standard datasets + multiple metrics + ablations
                  adequate= reasonable evaluation, some gaps
                  weak    = weak/insufficient evaluation
evidence_strength strong  = all main claims directly supported by reported results
                  adequate= mostly supported; minor over-claims
                  weak    = claims exceed evidence
novelty           strong  = clearly distinct contribution vs the corpus
                  adequate= incremental but real
                  weak    = marginal over prior corpus work
clarity           strong  = method reproducible from the text
                  adequate= mostly clear; some ambiguity
                  weak    = under-specified
```

### 11.2 Computed grade (code, not LLM)
```
points: strong = 1.0, adequate = 0.6, weak = 0.2, not_assessable = excluded
quality_score = round( mean(points over assessable dimensions), 2 )   # [0.2, 1.0]
grade: A if score ≥ 0.80 ; B if ≥ 0.55 ; C otherwise
```
`reproducibility` reads Phase 1's `git_exists`/`code_url` as ground truth for artifact availability — it is not re-derived by search (consistent with Phase 1 §8).

### 11.3 Depth handling
`abstract_only` papers can assess only dimensions supportable by the abstract; the rest are `not_assessable`. Their grade is computed over assessable dimensions and flagged `partial_quality_assessment`. Synthesis treats low-confidence/partial papers cautiously and says so.

---

## 12. Output Document Template

`02_literature_survey.md` follows this fixed structure. Per-paper records use a delimited, parseable block; synthesis sections are narrative but attribute every claim by `[paper_id]`.

```markdown
---
document_type: literature_survey
schema_version: "1.0"
generated_at: <ISO8601>
generator: anvesha.phase02_survey
source_phase1_dir: filtered_literature_v1_llm_agents_2023_2025
topic_statement: <string or null>
papers_total: <N>
papers_full_text: <n>
papers_abstract_only: <n>
metrics:
  coverage: <0–1>
  full_text_rate: <0–1>
  anchoring_rate: <0–1>
  synthesis_traceability: <0–1>
themes_total: <k>
---

# 02 — Literature Survey: <topic>

## 1. Objectives & Scope
<what this survey covers; corpus provenance (Phase 1 run, filters); what is out of scope>

## 2. Methodology
<extraction + quality + synthesis procedure; anchoring policy; reproducibility statement;
 note on depth (full_text vs abstract_only counts)>

## 3. Corpus Overview
<N papers; venue/year distribution; full-text vs abstract-only; thematic map summary>

## 4. Thematic Taxonomy
### Theme T1 — <name>
<definition> · Papers: [paper_001], [paper_004], ...
### Theme T2 — <name>
...

## 5. Per-Paper Summaries
<one block per paper, in rank order; the structured ExtractionRecord rendered as below>

### [paper_001] <Title> — <Authors, Year, Venue>  · quality: A · depth: full_text
- Problem: <...>
- Task: <...>
- Method: <...>
- Architecture: family=<...>; components=<...>; novel=<...>; base_model=<...>
- Datasets used: <name (role, task, modality, size)>; ...
- Benchmarks: <benchmark: metric=value (Δ vs baseline)>; ...
- Metrics: <...>
- Frameworks/Tools: <...>
- Key results: <...>
- Contributions: <...>
- Stated limitations: <...>
- Stated future work: <...>
- Relations: extends [paper_000]; contradicts [paper_007]; ...
- Code: git_exists=<bool>, code_url=<...>            (carried from Phase 1)

### [paper_002] ...

## 6. Datasets & Benchmarks Landscape
### 6.1 Usage Table
| Dataset/Benchmark | Modality | Task | Used by | Role(s) |
|---|---|---|---|---|
| <name> | <...> | <...> | [paper_001],[paper_005] | train/eval |
### 6.2 Dominant vs Emerging
<de-facto standards vs niche resources, with paper_ids>
### 6.3 Metric Conventions
<standard metrics per task>
### 6.4 Coverage Gaps            ← feeds Phase 3 (missing_benchmark)
<tasks/settings lacking shared benchmarks; single-paper benchmarks; thin regimes>

## 7. Frameworks & Architecture Analysis
### 7.1 Architecture Taxonomy
<families and shared components, with paper_ids>
### 7.2 Common Building Blocks
### 7.3 Backbone & Framework Trends

## 8. Comparative Synthesis
<like-for-like comparisons on shared benchmarks; trade-offs; tables with paper_ids>

## 9. Chronological Evolution & Trends
<how the field moved across the year range>

## 10. Consensus Findings
<agreed claims, each with supporting paper_ids; quality-weighted>

## 11. Contradictions & Open Debates     ← feeds Phase 3 (contradiction_between_papers)
<conflicting claims, with the opposing paper_ids and quoted positions>

## 12. Under-Explored Directions & Open Problems   ← feeds Phase 3
<aggregated future work + limitations + coverage/taxonomy gaps; each with motivating paper_ids>

## 13. Quality Assessment Summary
| paper_id | reproducibility | eval_rigor | evidence | novelty | clarity | grade |
|---|---|---|---|---|---|---|
<one row per paper; note partial assessments>
<corpus-level reliability statement>

## 14. Synthesis for Downstream Gap Analysis
<concise, structured recap that hands Phase 3 the raw material it consumes:
 - Open problems (from §12)
 - Contradictions (from §11)
 - Missing/!weak benchmarks (from §6.4)
 - Under-explored directions (from §12)
 each as a short bulleted list with paper_id attribution>
```

### 12.1 Output validity (exit criteria)
- Manifest present; `papers_total` equals the number of per-paper blocks; counts consistent.
- **Every** Phase 1 retained paper has a per-paper block (coverage = 1.0) or is explicitly listed as omitted with reason.
- Every per-paper block carries `quality` and `depth`.
- Every synthesis claim cites ≥ 1 `paper_id` (traceability check, Stage 6).
- Sections 11, 12, and 6.4 are present and non-empty when the corpus supports them (they are Phase 3's inputs).
- `anchoring_rate ≥ min_anchor_coverage` over full-text papers, or the shortfall is reported in the manifest.

---

## 13. Evaluation Metrics

Metrics that quantify whether the review was systematic and reproducible. Computed in Stage 8 and embedded in the manifest.

| Metric | Definition | Target |
|---|---|---|
| **Coverage** | fraction of Phase 1 retained papers with a completed extraction record | 1.00 |
| **Full-text rate** | fraction extracted from full PDF (vs abstract_only) | report; higher is better |
| **Anchoring rate** | fraction of extracted full-text claims with a *validated* `evidence_locator` | ≥ `min_anchor_coverage` (0.80) |
| **Synthesis traceability** | fraction of synthesis claims citing ≥ 1 `paper_id` | 1.00 |
| **Verification pass rate** | fraction of sampled synthesis claims confirmed supported (Stage 6) | ≥ 0.95 |
| **Quality-assessment completeness** | fraction of papers with a non-partial quality grade | report |
| **Dataset catalog completeness** | fraction of papers with ≥ 1 extracted dataset/benchmark (where applicable) | report |
| **Contradiction/gap yield** | counts populating §11/§12/§6.4 | report (informs Phase 3) |

These are quality metrics for the *survey process*, distinct from the evaluation *metrics used by the surveyed papers* (captured per-paper in §8 and summarized in §6.3).

---

## 14. Error Handling

| ID | Condition | Severity | Handling |
|---|---|---|---|
| EH-1 | Phase 1 output dir / index missing or unparseable | Fatal | Abort with a message pointing to Phase 1; no survey written. |
| EH-2 | A PDF is missing or yields no extractable text (incl. OCR failure) | Recoverable | Demote paper to `abstract_only`; extract from the index abstract; flag `abstract_only`; continue. |
| EH-3 | Extraction LLM call fails after retries for a paper | Recoverable | Emit a minimal record (metadata + abstract), flag `extraction_failed`; keep the paper; continue. |
| EH-4 | Anchor quote invalid (Stage 4) | Non-fatal | Field → `unverified`, quote discarded; counts toward anchoring-rate shortfall. |
| EH-5 | Quality assessment fails for a paper | Non-fatal | Grade omitted, flag `quality_unassessed`; paper retained. |
| EH-6 | Synthesis verification flags unsupported claims (Stage 6) | Non-fatal | Remove or qualify the claim; record in the verification log. |
| EH-7 | Corpus is empty (Phase 1 retained 0) | Non-fatal (guided) | Write a manifest-only survey noting the empty corpus and pointing back to Phase 1 filters. |
| EH-8 | Output write failure | Fatal | Abort; the last checkpoint allows `anvesha resume` to re-assemble. |
| EH-9 | Dataset/venue/framework name unresolved by alias map | Non-fatal | Keep verbatim; flag for the researcher to alias; aggregation still proceeds on the raw name. |

Principle: input/IO failures are fatal; per-paper failures are isolated and surfaced as explicit flags; no paper is silently dropped, and no unverifiable detail is presented as grounded.

---

## 15. Example Output (excerpt)

Illustrative — one per-paper block and two synthesis snippets (values fictional).

```markdown
### [paper_003] Self-RAG: Learning to Retrieve, Generate, and Critique — C. Author et al., 2024, ICML  · quality: A · depth: full_text
- Problem: retrieval-augmented LLMs retrieve indiscriminately and cannot assess their own output. <quote: "models retrieve a fixed number of passages regardless of need"> (Introduction, p.1)
- Task: open-domain question answering with on-demand retrieval.
- Method: train the model to emit reflection tokens that decide when to retrieve and to critique generated spans against retrieved evidence. (Method, p.3)
- Architecture: family=retrieval-augmented; components=[reflection-token controller, passage critic]; novel=[reflection tokens]; base_model=LLaMA-2-7B
- Datasets used: Natural Questions (eval, open-domain QA, text); ASQA (eval); training mix (pretrain). 
- Benchmarks: PopQA: accuracy=54.9 (Δ +8.1 vs baseline); TriviaQA: EM=66.4.
- Metrics: accuracy, EM, FactScore.
- Frameworks/Tools: PyTorch, HF Transformers.
- Key results: outperforms standard RAG and the base model on factuality benchmarks. <quote: "...improves factual accuracy without sacrificing fluency"> (Results, p.6)
- Contributions: reflection-token mechanism; self-critique training objective.
- Stated limitations: reflection adds inference overhead. <quote: "introduces additional decoding steps"> (Limitations, p.8)
- Stated future work: extend to multi-hop retrieval. <quote: "we leave multi-hop settings to future work"> (Conclusion, p.8)
- Relations: extends [paper_001]; compares_to [paper_005].
- Code: git_exists=true, code_url=https://github.com/example/self-rag

## 11. Contradictions & Open Debates (excerpt)
- **Does more retrieval help?** [paper_003] reports gains from *selective* retrieval and argues fixed top-k retrieval hurts factuality <quote: "fixed retrieval injects noise">, whereas [paper_008] reports monotonic gains from increasing retrieved passages <quote: "accuracy rises with k up to 20">. The corpus disagrees on the value of retrieval volume; no shared benchmark isolates the effect.

## 12. Under-Explored Directions & Open Problems (excerpt)
- **Multi-hop on-demand retrieval** — proposed as future work by [paper_003] and unaddressed by the selective-retrieval cluster ([paper_003],[paper_005]); no corpus paper evaluates selective retrieval in a multi-hop setting.
- **Cost-controlled evaluation** — retrieval-augmented methods report accuracy but rarely report inference cost ([paper_003] notes overhead but does not quantify it); no shared cost metric exists across the corpus.
```

---

## 16. Future Extensibility

Out of scope for v1.0; the design accommodates:
1. **Citation-graph expansion.** Optionally pull in highly-cited references of the corpus (via a discovery MCP) to contextualize — gated, since v1.0 surveys the fixed Phase 1 corpus.
2. **Emit machine records as data.** Optionally write a companion `02_literature_survey.records.json` with the raw `ExtractionRecord`s for programmatic consumers, alongside the human-readable MD.
3. **Quantitative meta-analysis.** Where many papers report the same benchmark+metric, compute aggregate statistics and effect sizes.
4. **Figure/table extraction.** Parse result tables and figures from PDFs to capture numbers not in prose (requires layout-aware extraction).
5. **Inter-rater reliability.** Run extraction twice with different model families and report agreement on key fields (dual-extraction), mirroring Phase 1's optional dual review.
6. **Configurable extraction template.** Expose the §7 field set so non-ML domains can adapt the extraction sheet.
7. **Cross-run survey diffing.** Compare a new survey version against a prior one to highlight what changed when the corpus is updated.

---

## Appendix A — Architecture Diagram

```
Phase 1 output dir:  01_filtered_literature.md  +  pdfs/paper_NNN.pdf
                                  │
                          ┌───────▼────────┐
                          │  Orchestrator  │  (stage sequencing; per-paper checkpoints; metrics)
                          └───────┬────────┘
   Stage 0   ┌───────────────────▼───────────────────┐
             │ Load & pair (index ↔ PDFs; set depth)  │
             └───────────────────┬───────────────────┘
   Stage 1   ┌───────────────────▼───────────────────┐
             │ PDF text extraction (PyMuPDF; OCR fb)  │  full_text papers
             └───────────────────┬───────────────────┘
   Stage 2   ┌───────────────────▼───────────────────┐  (LLM, per paper)
   (LLM)     │ Structured extraction (ExtractionRecord, anchored) │
             └───────────────────┬───────────────────┘
   Stage 3   ┌───────────────────▼───────────────────┐  (LLM, per paper)
   (LLM)     │ Quality assessment (rubric, anchored)  │
             └───────────────────┬───────────────────┘
   Stage 4   ┌───────────────────▼───────────────────┐
             │ Anchor validation (verbatim check)     │ → anchoring_rate
             └───────────────────┬───────────────────┘
   Stage 5   ┌───────────────────▼───────────────────┐  (LLM, over records)
   (LLM)     │ Cross-paper synthesis (taxonomy, datasets/benchmarks,
             │   architecture, comparison, trends, consensus,
             │   contradictions, under-explored directions)        │
             └───────────────────┬───────────────────┘
   Stage 6   ┌───────────────────▼───────────────────┐  (optional)
             │ Synthesis verification (traceability + support) │
             └───────────────────┬───────────────────┘
   Stage 7   ┌───────────────────▼───────────────────┐
             │ Survey assembly (§12 template)         │
             └───────────────────┬───────────────────┘
                                 ▼
                phases/02_literature_survey.md  →  INPUT to Phase 3 (Research Gaps)
```

---

## Appendix B — Internal Type Definitions

The emitted document structure is §12; these are the in-memory types.

```
SurveyState:
  source_phase1_dir   string
  papers              PaperWorkItem[]      rank-ordered
  taxonomy            Theme[]
  datasets_index      DatasetAgg[]         computed from DatasetUse across papers
  benchmark_index     BenchAgg[]
  synthesis           SynthesisBlocks      taxonomy/comparison/trends/consensus/contradictions/directions
  metrics             SurveyMetrics
  status              enum: "running" | "completed" | "failed"

PaperWorkItem:
  paper_id            string
  index_meta          (from Phase 1: title, authors, venue, year, doi, arxiv_id,
                       pdf_path, pdf_status, code_url, git_exists, relevance_score, rank, domain)
  full_text           string | null
  sections            {name → (text, page_start, page_end)} | null
  extraction_depth    enum: "full_text" | "abstract_only"
  record              ExtractionRecord     (§7.1)
  flags               string[]

ExtractionRecord, Field, Locator, Relation, DatasetUse, BenchResult,
ArchBlock, QualityAssessment, Level   — defined in §7, §8, §9, §11.

DatasetAlias / VenueAlias / FrameworkAlias:
  canonical_name      string
  aliases             string[]
  (DatasetAlias also: modality, typical_task)

SurveyMetrics:
  coverage, full_text_rate, anchoring_rate, synthesis_traceability,
  verification_pass_rate, quality_completeness, dataset_catalog_completeness   — floats
  contradiction_count, open_problem_count, coverage_gap_count                  — integers
```

---

## Appendix C — SDK Package Structure

Phase 2 lives under `anvesha/phases/phase02_survey/` and uses shared `core/` infrastructure identically to every phase (ANVESHA.md §7). It imports nothing from other phase modules and reads/writes only via the workspace module.

```
phase02_survey/
├── __init__.py
├── pipeline.py                 # LiteratureSurveyPipeline — Phase 2 entry point
│                               # accepts ResearchProject; runs Stages 0–8;
│                               # writes phases/02_literature_survey.md
├── orchestrator.py             # stage sequencing, per-paper checkpoints, metrics
├── state.py                    # SurveyState, PaperWorkItem
│
├── schemas/
│   ├── extraction.py           # ExtractionRecord, Field, Locator, Relation
│   ├── datasets.py             # DatasetUse, BenchResult, DatasetAgg, BenchAgg
│   ├── architecture.py         # ArchBlock
│   ├── quality.py              # QualityAssessment, Level, grade computation (§11.2)
│   └── survey_doc.py           # output template serializer (§12) + manifest
│
├── ingest/
│   ├── index_reader.py         # parse 01_filtered_literature.md (Phase 1 grammar)
│   ├── pdf_text.py             # PyMuPDF extraction + OCR fallback
│   └── sectioner.py            # IMRaD section segmentation + page offsets
│
├── extract/
│   ├── paper_extractor.py      # Stage 2 (LLM, anchored extraction)
│   ├── quality_assessor.py     # Stage 3 (LLM, rubric)
│   └── anchor_validator.py     # Stage 4 (verbatim quote validation)
│
├── synthesize/
│   ├── taxonomy.py             # Stage 5: thematic clustering
│   ├── datasets_landscape.py   # Stage 5: §8 aggregation + coverage gaps
│   ├── architecture_analysis.py# Stage 5: §9 analysis
│   ├── comparison.py           # Stage 5: like-for-like benchmark comparison
│   ├── trends.py               # Stage 5: chronological synthesis
│   ├── consensus_conflicts.py  # Stage 5: consensus + contradictions
│   └── directions.py           # Stage 5: under-explored directions / open problems
│
├── verify/
│   └── synthesis_verifier.py   # Stage 6 (traceability + support checks)
│
├── assemble/
│   └── survey_writer.py        # Stage 7 (§12 template) + Stage 8 metrics
│
└── aliases/
    ├── dataset_aliases.py      # DatasetAliasMap
    └── framework_aliases.py    # FrameworkAliasMap
```

Boundary compliance (ANVESHA.md §12): no cross-phase imports; all file I/O via `workspace/phase_io.py`; LLM access via injected client; entry point takes a `ResearchProject`.

---

## Appendix D — Shared Infrastructure (reference)

Phase 2 reuses Anvesha's shared infrastructure. See `ANVESHA.md`:

- **LLM adapters and backend selection** — §7.2, §10. Extraction/quality calls are per-paper and parallelizable; the **synthesis stage** is a single large-context operation over all records and benefits from a large-context model (Nemotron 3 Super, ~1M context — `research_gap_pipeline_implementation.md` §15.2). Greedy decoding throughout for determinism.
- **Checkpoint and resume** — §4.4. Phase 2 checkpoints per stage and per paper within extraction/quality; resume continues with un-extracted papers without re-reading completed ones.
- **Configuration and the `Phase` base class** — §7, §8. Inputs/options are injected via the loaded config; no stage reads config files directly.
- **Boundary rules** — §12 (identical hard constraints for all phases).
- **Versioning and `_VERSION_LOG.md`** — §11. Phase 2 emits a single file and auto-versions on re-run; the version log records the active survey for Phase 3.

---

*End of specification. Phase 2 (Literature Survey) v1.0.*
