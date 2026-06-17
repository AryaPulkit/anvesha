# Research Gap Analysis — Multi-Agent Pipeline
## Detailed Implementation Plan v1.2

> **Purpose of this document:** Step-by-step specification of every agent, schema, rubric, control flow rule, and feedback protocol. Written for direct use as a Claude Code implementation brief. Contains no code — only roles, data contracts, and logic rules.
>
> **This document covers Phase 3 (Research Gaps) of the Anvesha pipeline.** See `ANVESHA.md` for the full eleven-phase structure and project-wide context.
>
> **Changes in v1.2:**
> - Research Gaps is now **Phase 3** (a Filter Literature phase was added at position 1, pushing Survey to 2 and Gaps to 3)
> - Input paths updated: Literature MD is `02_literature_survey.md`; optional Approach MD is `03_approach.md`
> - Output path updated: `03_research_gaps.md`; Base Paper Profile output `04_base_paper.md`
> - SDK folder renamed `phase02_gaps/` → `phase03_gaps/`
> - Downstream workspace listing updated to the eleven-phase structure
>
> **Changes in v1.1:**
> - Workspace structure no longer references `PIPELINE.md` (consolidated into `ANVESHA.md`)
> - Phase 8 merged Evaluation & Error Analysis; Future Ideation and Loop Decision Gate renumbered
> - GapFinderOutput validation rules split for Refinement vs Discovery modes

---

## 1. System Overview

The pipeline takes two markdown files as input — a **Literature MD** (summarised research corpus) and an **Approach MD** (current methodology or system design) — and produces validated research gap–solution pairs as output. The Approach MD is **optional**: its presence or absence determines which of the two operating modes runs (see Section 5.2).

It runs in two sequential phases:

1. **Gap Discovery Phase** — finds, critiques, and validates research gaps
2. **Solution Generation Phase** — proposes, critiques, and validates implementation approaches for each validated gap

Each phase uses an identical three-tier review structure: Critic → Analyzer → LLM-as-Judge. Both Judges have access to a Web Search MCP tool for real-time verification.

---

## 2. Architecture Summary

```
[Literature MD] + [Approach MD (optional)]
        ↓
   [Orchestrator]
        ↓
╔══ GAP DISCOVERY LOOP ════════════════════════════════╗
║  [Gap Finder]                                        ║
║      ↓                                               ║
║  [Gap Critic × 3 personas]                           ║
║      ↓                                               ║
║  [Gap Analyzer] ──(tactical)──→ Gap Finder           ║
║      ↓ (escalate)                                    ║
╚══════════════════════════════════════════════════════╝
   [Gap LLM-as-Judge + Web Search]
        ├──(strategic + reset)──→ Gap Finder
        └──(APPROVE)──→ [Gap Prioritization]
                                ↓
                    [Base Paper Selection]
                    Agent shortlists candidates
                    Researcher makes final selection
                    Output: Base Paper Profile
                                ↓
╔══ SOLUTION LOOP (per gap) ═══════════════════════════╗
║  [Solution Generator]                                ║
║      ↓                                               ║
║  [Solution Critic × 3 personas]                      ║
║      ↓                                               ║
║  [Solution Analyzer] ──(tactical)──→ Generator       ║
║      ↓ (escalate)                                    ║
╚══════════════════════════════════════════════════════╝
   [Solution LLM-as-Judge + Web Search]
        ├──(strategic + reset)──→ Generator
        └──(APPROVE)──→ [Validated Output]
```

Instruction Merger sits between Judge/Analyzer outputs and the Generator entry point in both loops. It is triggered only when both tactical and strategic instructions are simultaneously active.

---

## 3. Core Design Principles

These are deliberate architectural decisions made upfront to avoid well-known failure modes in multi-agent pipelines. They are baked into the design from the start — not retrofits.

### Decision 1 — Different Model Families for Generator and Critic
The Critic must use a different model family from the Generator. In this pipeline: Generator uses Nemotron 3 Super (NVIDIA), Critic uses GPT-OSS-120B (OpenAI). These have different architectures, different training data, and different pre-training approaches, meaning their blind spots do not correlate. A Critic built on the same model as the Generator tends to miss exactly what the Generator missed.

### Decision 2 — Rubric-Based Analyzer
The Analyzer does not make open-ended "is this critique valid?" judgments. It scores every Critic comment against a **four-criterion rubric**. Only comments passing ≥ 3 of 4 criteria are forwarded as instructions. This replaces unconstrained LLM reasoning with structured classification, which is significantly more reliable for a gatekeeping role.

### Decision 3 — Evidence Anchoring on Critic
Every critique claim must include a **verbatim citation** from the source documents (Literature MD or Approach MD). A critique without an anchor citation is automatically rejected by the Analyzer regardless of rubric scores. This makes critique validation tractable, auditable, and resistant to hallucinated critique claims.

### Decision 4 — Sequential Authority Protocol
When the Judge issues strategic feedback, it sets a `supersedes_tactical` flag. If `true`, the Analyzer loop resets and the Generator receives only the Judge's instruction. This ensures the Generator never receives conflicting instructions from two different authority levels at the same time.

### Decision 5 — Multi-Angle Critic (3 Personas)
Each Critic runs as three independent persona instances with distinct evaluation lenses:
- **Persona A — Methodological:** Research rigor, validity, evidence quality
- **Persona B — Domain Expert:** Domain relevance, literature alignment, novelty
- **Persona C — Feasibility / Coverage:** Practical viability, scope, completeness

Critiques appearing in 2 or more personas are flagged as **consensus critiques** and treated as high priority. A single-persona critic is more likely to miss issues outside its natural framing.

### Decision 6 — Instruction Merger
When both the Analyzer (tactical) and Judge (strategic) have active instructions for the same Generator pass, a dedicated **Instruction Merger** step consolidates them into one coherent, conflict-free instruction set. The Generator always receives a single instruction, never two competing ones.

### Decision 7 — Adaptive Web Search on Judge
Both Judges run an iterative web search loop (up to 3 rounds) before scoring. Each round's queries are informed by findings from the previous round, allowing the Judge to drill into specifics rather than running all queries blindly upfront. This grounds final decisions in real-world evidence and addresses model knowledge cutoff limitations.

### Decision 8 — Mode-Aware Gap Finder
The pipeline supports two distinct research modes detected automatically from the inputs:
- **Refinement Mode** — researcher provides an Approach MD; gaps are found between what the field knows and what the approach does. This is *improvement research*.
- **Discovery Mode** — no Approach MD; gaps are open problems in the field itself (contradictions, missing benchmarks, unexplored directions). This is *original-contribution research*.

The mode is detected from input presence — the researcher does not configure it manually. Validation rules, evidence categorisation, and system prompts differ per mode (see Section 5.2).

---

## 4. Global Type Definitions

These types are shared across agents. All agent schemas reference these.

### 4.1 EvidenceAnchor
```
field: anchor_quote       type: string    constraint: exact verbatim text from source document, 10–300 chars
field: anchor_source      type: enum      values: "literature_md" | "approach_md" | "generator_output"
field: anchor_section     type: string    description: section heading or paragraph identifier
field: anchor_relevance   type: string    description: one sentence explaining why this quote supports the claim
```

### 4.2 IterationState
```
field: iteration_number          type: integer   description: current loop iteration count (1-indexed)
field: tactical_iteration_count  type: integer   description: iterations triggered by Analyzer
field: strategic_iteration_count type: integer   description: iterations triggered by Judge
field: last_instruction_source   type: enum      values: "analyzer" | "judge" | "merger" | "none"
field: judge_supersede_active    type: boolean   description: true if Judge's last instruction had supersedes_tactical=true
```

### 4.3 RubricScore
```
field: anchored        type: boolean   description: critique has a valid EvidenceAnchor
field: specific        type: boolean   description: critique identifies a concrete, named flaw (not vague)
field: actionable      type: boolean   description: critique can be turned into a specific one-sentence instruction
field: non_redundant   type: boolean   description: critique is not a duplicate of another critique in this batch
field: total_passed    type: integer   range: 0–4
field: is_valid        type: boolean   rule: total_passed >= 3
```

### 4.4 WebSearchRecord
A single executed query and its result. Used inside a SearchRound.
```
field: query            type: string    description: exact query string submitted
field: purpose          type: enum      values: "verify_novelty" | "validate_citation" | "check_prior_work" | "check_approach_precedent" | "check_known_failures" | "follow_up_on_finding"
field: key_finding      type: string    description: 1–3 sentence summary of what was found
field: source_url       type: string    description: primary source URL returned
field: result_relevance type: enum      values: "highly_relevant" | "partially_relevant" | "not_relevant"
field: influenced_score type: enum      values: "novelty_score" | "evidence_score" | "feasibility_score" | "none"
```

### 4.5 SearchRound
One round of search activity within the Judge's adaptive search loop. The Judge may execute up to 3 rounds. Each round contains 1–2 queries, followed by a self-assessment of whether enough context has been gathered to score confidently.
```
field: round_number           type: integer   range: 1–3
field: queries                type: WebSearchRecord[]   constraint: 1–2 queries per round
field: round_findings_summary type: string    description: 2–4 sentences synthesising what this round established
field: context_assessment
  field: confidence_to_score  type: enum      values: "low" | "medium" | "high"
    description:
      "high"   = search has resolved the key unknowns; no further search needed
      "medium" = sufficient context to score with moderate confidence; stop if round >= 2
      "low"    = critical questions remain unresolved; continue if rounds remain
  field: unresolved_questions type: string[]  description: specific open questions driving the next round; empty if confidence is high
  field: continue_searching   type: boolean
    rule: true only if confidence_to_score = "low" AND round_number < MAX_SEARCH_ROUNDS AND queries_remaining > 0
  field: stop_reason          type: string | null   description: why search stopped early
```

### 4.6 AdaptiveSearchState
Tracks the full search session for a single Judge invocation.
```
field: rounds                 type: SearchRound[]
field: total_queries_run      type: integer
field: total_rounds_used      type: integer
field: search_terminated_by   type: enum      values: "high_confidence" | "max_rounds" | "no_relevant_results" | "queries_exhausted" | "skipped"
field: final_search_summary   type: string    description: 3–6 sentences on what the full search session established and how it informed scoring
```

---

## 5. Agent Specifications

---

### 5.1 Orchestrator

**Role:** Manages pipeline-wide state. Routes inputs to agents, maintains iteration counters, enforces hard limits, tracks the status of every gap through both loops. Not an LLM reasoning agent — a stateful controller.

**Model:** Rule-based controller (no LLM call required at the orchestration layer).

**Pipeline State Schema:**
```
field: pipeline_run_id              type: string
field: created_at                   type: ISO timestamp
field: inputs
  field: literature_md              type: string
  field: approach_md                type: string | null   description: null in Discovery Mode
  field: mode                       type: enum: "refinement" | "discovery"
field: gap_discovery
  field: gap_finder_raw_output      type: GapFinderOutput
  field: gaps                       type: GapReviewState[]
  field: prioritized_gap_ids        type: string[]
field: solution_generation
  field: solutions                  type: SolutionReviewState[]
field: final_outputs                type: ValidatedOutput[]
field: status                       type: enum: "running" | "completed" | "failed" | "partial"

GapReviewState:
  field: gap_id                     type: string
  field: iteration_state            type: IterationState
  field: critic_outputs             type: GapCriticOutput[]
  field: analyzer_outputs           type: GapAnalyzerOutput[]
  field: judge_outputs              type: GapJudgeOutput[]
  field: current_instruction        type: MergedInstruction | null
  field: status                     type: enum: "pending" | "in_review" | "approved" | "rejected" | "unresolvable"

SolutionReviewState:
  field: gap_id                     type: string
  field: iteration_state            type: IterationState
  field: generator_outputs          type: SolutionGeneratorOutput[]
  field: critic_outputs             type: SolutionCriticOutput[]
  field: analyzer_outputs           type: SolutionAnalyzerOutput[]
  field: judge_outputs              type: SolutionJudgeOutput[]
  field: current_instruction        type: MergedInstruction | null
  field: status                     type: enum: "pending" | "in_review" | "approved" | "rejected" | "unresolvable"
```

---

### 5.2 Gap Finder Agent

**Role:** Reads input documents and identifies research gaps. Operates in one of two modes depending on whether the researcher has an existing approach. The mode is detected automatically from the inputs — the researcher does not configure it manually.

**Model:** Nemotron 3 Super (Generator model)

---

**Mode Detection:**
```
IF approach_md is present and non-empty → Refinement Mode
IF approach_md is absent or empty       → Discovery Mode
```

---

**Refinement Mode**
*The researcher has existing work. Gaps are found between what the field knows and what the researcher's current approach does.*

Inputs:
```
field: literature_md          type: string   required
field: approach_md            type: string   required (non-empty)
field: iteration_instruction  type: string | null
field: iteration_state        type: IterationState
field: mode                   type: string   value: "refinement"
```

System Prompt Guidance (Refinement Mode):
- Compare Literature MD and Approach MD section by section
- For each gap: provide a verbatim quote from the Literature MD showing what the field addresses, and a direct statement from the Approach MD showing absence, contradiction, or incomplete coverage
- Identify four gap types relative to the researcher's approach: what the approach misses entirely, what it contradicts, what it approximates but insufficiently, and what the field considers standard that the approach lacks
- Severity is rated relative to how much the gap undermines the approach's validity or completeness
- Do not invent gaps — every gap must be grounded in both documents
- If `iteration_instruction` is present, address every point before adding new content

---

**Discovery Mode**
*The researcher is starting fresh. Gaps are open problems found within the literature itself — no existing approach to compare against.*

Inputs:
```
field: literature_md          type: string   required
field: approach_md            type: null     absent or empty
field: iteration_instruction  type: string | null
field: iteration_state        type: IterationState
field: mode                   type: string   value: "discovery"
```

System Prompt Guidance (Discovery Mode):
- Analyse the Literature MD as a standalone corpus — do not assume any existing approach
- Identify gaps within the field itself across four categories:
    Open problems: problems stated as unsolved or as future work across multiple papers
    Contradictions: conflicting results or claims between papers that remain unresolved
    Missing benchmarks: evaluation gaps — metrics, datasets, or baselines absent from the field
    Unexplored directions: promising directions mentioned but never pursued in depth
- For each gap: cite at least two papers that independently acknowledge or imply the gap
- Rate severity based on how widely the gap is acknowledged and how much it limits the field's progress
- Do not frame gaps relative to any specific system — frame them as field-level open problems
- If `iteration_instruction` is present, address every point before adding new content

---

**Output Schema — GapFinderOutput:**
```
field: run_id                 type: string
field: gap_id_prefix          type: string
field: iteration_number       type: integer
field: mode                   type: enum: "refinement" | "discovery"
field: gaps                   type: Gap[]
field: total_gaps_found       type: integer
field: analysis_summary       type: string   constraint: 100–300 words

Gap:
  field: gap_id               type: string   format: GAP-{3-digit number}
  field: title                type: string   constraint: max 12 words
  field: description          type: string   constraint: 100–200 words
  field: gap_type             type: enum     values: "missing_coverage" | "contradiction" | "missing_benchmark" | "unexplored_subproblem" | "outdated_approach"
  field: severity             type: enum     values: "critical" | "high" | "medium" | "low"
  field: novelty_hypothesis   type: string
  field: evidence             type: GapEvidence[]   constraint: minimum 2 items

GapEvidence:
  field: claim                type: string
  field: anchor               type: EvidenceAnchor
  field: evidence_role        type: enum
    Refinement Mode values:
      "shows_approach_absence"          literature covers this; approach does not
      "shows_approach_contradiction"    approach contradicts what literature establishes
      "shows_approach_insufficiency"    approach partially covers but inadequately
      "shows_missing_standard"          field standard that the approach lacks
    Discovery Mode values:
      "open_problem_stated"             paper explicitly states this as unsolved/future work
      "contradiction_between_papers"    two papers make conflicting claims on this point
      "missing_benchmark"               evaluation gap acknowledged across papers
      "unexplored_direction"            direction mentioned but never pursued in depth
```

**Validation Rules (mode-aware):**

*Refinement Mode (approach_md present):*
- Every gap must have ≥ 1 evidence item with `anchor_source: "literature_md"`
- Every gap must have ≥ 1 evidence item with `anchor_source: "approach_md"`
- Every `evidence_role` must use a Refinement Mode enum value
- A gap with no `approach_md`-anchored evidence is invalid (suggests the gap is field-level, not approach-specific)

*Discovery Mode (approach_md absent):*
- Every gap must have ≥ 2 evidence items, all with `anchor_source: "literature_md"`
- The two literature-anchored evidence items should cite different papers where possible (single-paper gaps are flagged as low-confidence)
- Every `evidence_role` must use a Discovery Mode enum value
- A gap with any `approach_md`-anchored evidence is invalid (no approach exists)

*Both modes:*
- `severity: "critical"` requires minimum 3 evidence items
- `GapFinderOutput.mode` must match the orchestrator's detected mode (rejected if mismatched)

---

### 5.3 Gap Critic Agent (Multi-Angle)

**Role:** Adversarially evaluates the Gap Finder's output. Runs as three independent personas. Has no stop/continue logic. Only produces structured critique grounded in source citations.

**Model:** GPT-OSS-120B (different family from Generator — Decision 1)

**Personas:**

**Persona A — Methodological Critic**
- Checks: Is the gap claim methodologically valid? Is the evidence logically sound? Does cited literature actually support the stated gap?
- Critique types: `evidence_misread` | `logical_leap` | `insufficient_evidence` | `overgeneralisation`

**Persona B — Domain Expert Critic**
- Checks: Is this gap truly novel or already addressed? Is it relevant to the domain? Is severity rating justified?
- Critique types: `gap_already_addressed` | `gap_irrelevant_to_domain` | `severity_overstated` | `severity_understated` | `novelty_questionable`

**Persona C — Scope & Coverage Critic**
- Checks: Is the gap scoped correctly? Too broad? Too narrow? Does description align with evidence?
- Critique types: `scope_too_broad` | `scope_too_narrow` | `description_evidence_mismatch` | `gap_type_miscategorised`

**Output Schema — GapCriticOutput (one per persona per gap):**
```
field: run_id                 type: string
field: gap_id                 type: string
field: critic_persona         type: enum     values: "methodological" | "domain_expert" | "scope_coverage"
field: iteration_number       type: integer
field: critiques              type: GapCritique[]
field: overall_gap_assessment type: enum     values: "valid_as_stated" | "valid_with_refinement" | "questionable" | "invalid"
field: assessment_rationale   type: string   constraint: 50–150 words
field: confidence             type: float    range: 0.0–1.0

GapCritique:
  field: critique_id          type: string   format: CRIT-{persona_initial}-{number}
  field: critique_type        type: enum     values: as listed per persona above
  field: claim                type: string
  field: anchor               type: EvidenceAnchor   constraint: REQUIRED
  field: severity             type: enum     values: "fatal" | "major" | "minor"
  field: suggested_correction type: string
  field: affects_gap_ids      type: string[]
```

**Validation Rules:**
- A GapCritique with no `anchor` field populated is unanchored and rejected by Analyzer regardless of rubric score
- `severity: "fatal"` must have `suggested_correction` of at least 30 words

---

### 5.4 Gap Analyzer Agent

**Role:** Receives all three Critic persona outputs for a given gap. Applies the rubric to every critique. Filters noise. Identifies consensus critiques. Decides: iterate or escalate to Gap Judge.

**Model:** Any capable model (rubric constrains open-ended reasoning significantly)

**Rubric — Applied to every GapCritique:**
```
Criterion 1 — Anchored
  Pass: anchor_quote is non-empty AND anchor_source is valid AND anchor_section is non-empty
  Fail: anchor missing or anchor_quote is paraphrase not verbatim

Criterion 2 — Specific
  Pass: critique_type is set AND claim describes a specific named issue
  Fail: claim uses vague language without specific reference

Criterion 3 — Actionable
  Pass: suggested_correction describes a specific change the Gap Finder can make
  Fail: suggested_correction is abstract advice without specific direction

Criterion 4 — Non-Redundant
  Pass: the core concern is not covered by another critique already marked valid
  Fail: two critiques making the same point about the same gap evidence

Validity threshold: total_passed >= 3 → is_valid = true
```

**Consensus Detection:** Critiques from 2+ personas with same `critique_type` pointing to same `gap_id` → `is_consensus: true`. Consensus critiques escalated to priority regardless of severity.

**Decision Rules:**
```
IF no valid critiques → decision = "escalate_to_judge"
IF valid critiques AND tactical_iteration_count < MAX_TACTICAL_ITERATIONS → decision = "iterate"
IF valid critiques AND tactical_iteration_count >= MAX_TACTICAL_ITERATIONS → decision = "escalate_to_judge"
```

**Output Schema — GapAnalyzerOutput:**
```
field: run_id                     type: string
field: gap_id                     type: string
field: iteration_number           type: integer
field: scored_critiques           type: ScoredCritique[]
field: valid_critique_count       type: integer
field: consensus_critique_count   type: integer
field: noise_critique_count       type: integer
field: decision                   type: enum: "iterate" | "escalate_to_judge"
field: iteration_instruction      type: string | null
field: escalation_reason          type: string | null
field: top_unresolved_issues      type: string[]

ScoredCritique:
  field: critique_id              type: string
  field: source_persona           type: string
  field: rubric_score             type: RubricScore
  field: is_consensus             type: boolean
  field: consensus_personas       type: string[]
  field: included_in_instruction  type: boolean
```

---

### 5.5 Gap LLM-as-Judge

**Role:** Receives the full iteration history for a gap and makes the final approval decision. Runs an adaptive web search loop before scoring. Issues strategic instructions when continuing. Is the only agent with authority to approve a gap for the solution loop.

**Model:** Strongest available model (e.g., Claude Opus 4.6)

**Web Search MCP — Adaptive Search Loop:**
```
ADAPTIVE SEARCH LOOP — Gap Judge

Constraints:
  MAX_SEARCH_ROUNDS      = 3
  MAX_QUERIES_PER_ROUND  = 2
  MAX_TOTAL_QUERIES      = 5

Round 1 — Broad orientation queries:
  Purpose: establish whether the gap domain has relevant recent literature
  Query types: "verify_novelty", "check_prior_work"
  After Round 1: assess confidence_to_score
    "high"   → stop, proceed to scoring
    "medium" AND round >= 2 → stop
    "low"    → continue to Round 2

Round 2 — Targeted follow-up (only if Round 1 confidence = "low"):
  Purpose: resolve specific open questions from Round 1
  Query types: "validate_citation", "verify_novelty", "follow_up_on_finding"
  Rule: each query must reference a specific entity from Round 1 findings
  After Round 2: if confidence = "high" or "medium" → stop

Round 3 — Specific verification only (rare):
  Max queries: 1
  Query type: "validate_citation" or "follow_up_on_finding" only
  After Round 3: always stop regardless of confidence
```

**Early Stop Conditions:**
- Round 1 returns no relevant results → stop, `search_terminated_by = "no_relevant_results"`
- Any round achieves `confidence_to_score = "high"` → stop immediately
- `total_queries_run` reaches 5 → stop immediately
- Previous Judge for this gap already found no results → skip entirely

**Quality Assessment Rubric:**
```
novelty_score (0–10):
  10 = gap is clearly novel, web search confirms not addressed
  7–9 = likely novel, minor related work exists but does not close it
  4–6 = partially addressed in literature; refinement needed
  1–3 = substantially addressed; strong prior work found
  0 = fully solved in existing literature; reject

evidence_quality_score (0–10):
  10 = all anchors accurate, citations verified, logic tight
  7–9 = most anchors valid, minor issues
  4–6 = some anchors weak or paraphrased
  1–3 = evidence largely misrepresents sources
  0 = no valid evidence; reject

gap_clarity_score (0–10):
  10 = precisely scoped, clearly defined, unambiguous
  7–9 = clear with minor scope issues
  4–6 = vague or too broad/narrow
  1–3 = poorly defined, hard to translate into solution
  0 = cannot be acted upon; reject

overall_score = (novelty_score × 0.4) + (evidence_quality_score × 0.35) + (gap_clarity_score × 0.25)
```

**Decision Rules:**
```
overall_score >= 7.0 → APPROVE
overall_score >= 5.0 AND strategic_iteration_count < MAX_STRATEGIC_ITERATIONS → CONTINUE
overall_score < 5.0 AND strategic_iteration_count >= MAX_STRATEGIC_ITERATIONS → REJECT
overall_score < 5.0 AND strategic_iteration_count < MAX_STRATEGIC_ITERATIONS → CONTINUE
```

**Output Schema — GapJudgeOutput:**
```
field: run_id                     type: string
field: gap_id                     type: string
field: iteration_number           type: integer
field: search_session             type: AdaptiveSearchState
field: web_search_informed        type: boolean
field: quality_assessment
  field: novelty_score            type: float   range: 0–10
  field: evidence_quality_score   type: float   range: 0–10
  field: gap_clarity_score        type: float   range: 0–10
  field: overall_score            type: float   range: 0–10
  field: score_rationale          type: string  constraint: 100–200 words
field: decision                   type: enum: "APPROVE" | "CONTINUE" | "REJECT"
field: supersedes_tactical        type: boolean
field: strategic_instruction      type: string | null
field: approval_rationale         type: string | null
field: rejection_rationale        type: string | null
field: search_influence_summary   type: string
```

**Supersedes Tactical Rule:**
- Direction change (new framing, different evidence base, scope shift) → `supersedes_tactical = true`
- Additive instruction (strengthen evidence, minor clarity) → `supersedes_tactical = false`

---

### 5.6 Instruction Merger (Gap Loop)

**Role:** Triggered only when both a tactical instruction (from Gap Analyzer) and a strategic instruction (from Gap Judge) are simultaneously active. Produces one coherent, prioritised instruction.

**Model:** Any capable model.

**When triggered:**
- Gap Judge issued CONTINUE with `supersedes_tactical = false`
- AND Gap Analyzer has an active `iteration_instruction`

**Merge Rules:**
1. Remove tactical points that contradict the strategic instruction
2. Keep tactical points compatible with and subordinate to strategic direction
3. If tactical addresses same issue as strategic, use strategic formulation
4. Merged instruction: strategic direction first, tactical refinements second

**Output Schema — MergedInstruction:**
```
field: gap_id                     type: string
field: tactical_instruction_in    type: string
field: strategic_instruction_in   type: string
field: merged_instruction         type: string   constraint: max 200 words
field: conflicts_detected         type: boolean
field: conflict_resolution        type: string | null
field: tactical_points_retained   type: integer
field: tactical_points_dropped    type: integer
field: priority_basis             type: enum: "strategic_only" | "tactical_only" | "merged"
```

---

### 5.7 Gap Prioritization Step

**Role:** Runs once after all gaps are approved by the Gap Judge. Ranks approved gaps by impact and feasibility before the Solution Loop begins.

**Model:** Any capable model (lightweight task).

**Scoring Rubric:**
```
impact_score (0–10):
  10 = closing this gap fundamentally changes what the current approach can do
  7–9 = significant improvement to outcomes or coverage
  4–6 = moderate improvement
  1–3 = minor improvement
  0 = negligible

feasibility_score (0–10):
  10 = clear path to implementation with existing tools/resources
  7–9 = feasible with moderate effort
  4–6 = requires significant new work
  1–3 = high uncertainty
  0 = currently infeasible

priority_score = (impact_score × 0.6) + (feasibility_score × 0.4)
```

**Output Schema — PrioritizationOutput:**
```
field: prioritized_gaps           type: PrioritizedGap[]
field: top_n_selected             type: string[]
field: excluded_gaps              type: string[]
field: exclusion_rationale        type: object   key: gap_id, value: reason string

PrioritizedGap:
  field: gap_id                   type: string
  field: rank                     type: integer
  field: impact_score             type: float
  field: feasibility_score        type: float
  field: priority_score           type: float
  field: impact_rationale         type: string   constraint: 30–80 words
  field: feasibility_rationale    type: string   constraint: 30–80 words
```

---

### 5.8 Base Paper Selection Agent

**Role:** Given the approved and prioritised gaps, identifies candidate base papers from the Literature MD and produces a structured shortlist for the researcher to choose from. This is the only agent in the pipeline with a **mandatory human decision gate** — the agent shortlists and profiles, the researcher selects. The agent then produces a Base Paper Profile from the selected paper, which feeds into the Solution Generator.

**Model:** Nemotron 3 Super

**When it runs:** Once, after Gap Prioritization and before the Solution Loop. Operates on the full prioritised gap list, not per-gap.

**Why a human gate is mandatory here:**
Base paper selection determines the entire framing of the research contribution. A wrong base paper — one that misaligns with the researcher's actual intent — propagates through every downstream phase. No agent has enough context about the researcher's goals, institutional constraints, or personal research direction to make this call reliably. The agent's job is to reduce the decision from hundreds of papers to a shortlist of 3–5 with clear rationale. The researcher makes the final call in under 10 minutes.

---

**Step 1 — Candidate Shortlisting**

Inputs:
```
field: literature_md              type: string
field: prioritised_gaps           type: PrioritizedGap[]
field: gap_finder_mode            type: enum: "refinement" | "discovery"
field: approach_md                type: string | null  (present in Refinement Mode only)
```

System Prompt Guidance (Step 1):
- For each top-ranked gap, identify the 2–3 papers in the Literature MD that most directly address or relate to it
- Score each candidate paper against all five criteria (defined below)
- A paper may appear as a candidate for multiple gaps — deduplicate and note which gaps it covers
- Produce a ranked shortlist of 3–5 unique candidates across all top gaps
- Do not recommend papers not present in the Literature MD

**Scoring Criteria:**
```
Criterion 1 — Gap Relevance (0–10)
  Does this paper directly address the identified gap, or represent
  the closest existing attempt at solving it?
  10 = paper is the primary existing work on this exact gap
  7–9 = paper is highly relevant, covers most of the gap
  4–6 = paper is related but only partially relevant
  1–3 = tangentially related; weak connection to the gap
  0 = not relevant to any prioritised gap

Criterion 2 — Methodological Alignment (0–10)
  Would building on or challenging this paper's method constitute
  a natural and valid research contribution?
  10 = method is directly extendable or challengeable in a clear direction
  7–9 = method is relevant; extension path exists with some adaptation
  4–6 = method is useful as reference but not a direct base
  1–3 = method is too distant to build on directly
  0 = no methodological connection

Criterion 3 — Recency and Field Standing (0–10)
  Is this paper current state-of-the-art, or has it been superseded?
  Is it actively cited and extended by the field?
  10 = current SotA, actively cited, live conversation in the field
  7–9 = recent, well-cited, still relevant
  4–6 = older but foundational; field still references it
  1–3 = largely superseded; cited mainly for historical context
  0 = obsolete or retracted

Criterion 4 — Reproducibility (0–10)
  Can the researcher actually build on this paper?
  10 = code available, dataset accessible, method fully described
  7–9 = most artifacts available; minor gaps
  4–6 = partial availability; some reconstruction needed
  1–3 = significant reproducibility barriers
  0 = no artifacts; method underspecified; not buildable

Criterion 5 — Contribution Clarity (0–10)
  How clearly does this paper define what "improving on it" would mean?
  Does it state its own limitations or future directions explicitly?
  10 = limitations clearly stated; improvement directions obvious
  7–9 = some limitations noted; improvement path clear
  4–6 = limitations implied but not explicit; path requires inference
  1–3 = paper claims completeness; improvement path unclear
  0 = no limitations stated; contribution path opaque

composite_score = (gap_relevance × 0.30) + (methodological_alignment × 0.25)
               + (recency_standing × 0.20) + (reproducibility × 0.15)
               + (contribution_clarity × 0.10)
```

**Step 1 Output Schema — BasePaperShortlist:**
```
field: run_id                     type: string
field: gaps_considered            type: string[]  gap_ids of top-ranked gaps used
field: candidates                 type: BasePaperCandidate[]  (3–5 items, ranked)
field: shortlist_rationale        type: string    constraint: 100–200 words overall summary

BasePaperCandidate:
  field: candidate_id             type: string    format: BP-{number}
  field: paper_title              type: string
  field: paper_anchor             type: EvidenceAnchor  citation from Literature MD
  field: relevant_gaps            type: string[]  which gap_ids this paper addresses
  field: scores
    field: gap_relevance          type: float   range: 0–10
    field: methodological_alignment type: float range: 0–10
    field: recency_standing       type: float   range: 0–10
    field: reproducibility        type: float   range: 0–10
    field: contribution_clarity   type: float   range: 0–10
    field: composite_score        type: float   range: 0–10
  field: what_building_on_means   type: string  constraint: 50–100 words
                                  description: concrete description of what a research
                                  contribution building on this paper would look like
  field: key_limitation           type: string  constraint: 30–60 words
                                  description: the most significant limitation the paper
                                  acknowledges or that the field has identified
  field: artifacts_available      type: string  description: code, dataset, model weights status
```

---

**Step 2 — Human Decision Gate**

The Orchestrator pauses the pipeline and presents the shortlist to the researcher. This is not an optional step — the pipeline does not proceed until the researcher makes a selection.

```
HUMAN GATE PROTOCOL:

Orchestrator presents:
  — shortlist of 3–5 candidates with scores and rationale
  — what_building_on_means for each
  — key_limitation for each

Researcher options:
  a) Select one candidate → pipeline continues with that paper
  b) Select none and provide a paper title not in the shortlist
     → agent produces profile for researcher-specified paper
     → pipeline continues with that paper
  c) Request a deeper search → agent runs a second shortlisting pass
     with different criteria weights specified by researcher
  d) Abort phase → researcher reviews Literature MD before proceeding

Timeout: none — researcher takes as long as needed
The checkpoint system saves state; the researcher can close the
session and return later to make the selection.
```

**Human gate output (researcher records this):**
```
field: selected_candidate_id      type: string   "BP-{number}" or "researcher-specified"
field: selected_paper_title       type: string
field: selection_rationale        type: string   researcher's own note (optional)
field: selected_at                type: ISO timestamp
```

---

**Step 3 — Base Paper Profile**

Once the researcher has selected, the agent produces a structured profile of the chosen paper. This profile is injected as context into the Solution Generator and Idea Generation phases.

Inputs:
```
field: selected_paper             type: BasePaperCandidate (or researcher-specified title)
field: literature_md              type: string
field: prioritised_gaps           type: PrioritizedGap[]
```

System Prompt Guidance (Step 3):
- Extract a complete structured profile of the selected paper from the Literature MD
- Summarise the method in enough detail that the Solution Generator can build on it
- List all limitations stated by the authors and identified by citing papers
- List all future directions mentioned by the authors
- Identify which of the prioritised gaps this paper most directly relates to
- Note what the paper explicitly does not attempt

**Step 3 Output Schema — BasePaperProfile:**
```
field: paper_title                type: string
field: paper_anchor               type: EvidenceAnchor
field: method_summary             type: string   constraint: 150–300 words
field: core_contribution          type: string   constraint: 50–100 words
field: datasets_used              type: string[]
field: metrics_reported           type: string[]
field: baselines_compared         type: string[]
field: author_stated_limitations  type: string[]  verbatim or close paraphrase with anchor
field: author_stated_future_work  type: string[]  verbatim or close paraphrase with anchor
field: field_identified_gaps      type: string[]  limitations noted by papers that cite this one
field: artifacts
  field: code_available           type: boolean
  field: code_url                 type: string | null
  field: dataset_available        type: boolean
  field: dataset_url              type: string | null
  field: model_weights_available  type: boolean
field: relevant_to_gaps           type: string[]  gap_ids from prioritised list
field: does_not_address           type: string[]  explicit scope exclusions from the paper
```

---

### 5.9 Solution Generator Agent

**Role:** For each prioritised gap, proposes a concrete implementation approach. On first pass generates fresh. On subsequent passes incorporates instruction from Analyzer or Judge (via Merger if both active).

**Model:** Nemotron 3 Super

**Output Schema — SolutionGeneratorOutput:**
```
field: run_id                     type: string
field: gap_id                     type: string
field: approach_id                type: string  format: SOLN-{gap_id}-{iteration}
field: iteration_number           type: integer
field: approach_title             type: string  constraint: max 12 words
field: approach_summary           type: string  constraint: 100–150 words
field: gap_closure_statement      type: string  constraint: 50–100 words
field: implementation_steps       type: ImplementationStep[]  constraint: 3–8 steps
field: known_limitations          type: string[]  constraint: minimum 2
field: success_criteria           type: string[]  constraint: minimum 3 measurable criteria
field: literature_alignment       type: LiteratureAlignment[]

ImplementationStep:
  field: step_id                  type: string  format: STEP-{number}
  field: step_title               type: string  constraint: max 8 words
  field: step_description         type: string  constraint: 80–200 words
  field: required_resources       type: string[]
  field: complexity               type: enum: "low" | "medium" | "high"
  field: evidence_support         type: EvidenceAnchor[]  constraint: minimum 1 per step

LiteratureAlignment:
  field: approach_element         type: string
  field: literature_support       type: EvidenceAnchor
  field: alignment_type           type: enum: "directly_supported" | "analogously_supported" | "inferred" | "novel_extension"
```

---

### 5.10 Solution Critic Agent (Multi-Angle)

**Role:** Adversarially evaluates the Solution Generator's proposed approach. Runs as three independent personas. No stop/continue logic. All critiques must be grounded in citations.

**Model:** GPT-OSS-120B

**Personas:**

**Persona A — Methodological Critic**
- Critique types: `methodological_flaw` | `logical_gap_in_steps` | `citation_misrepresented` | `unsupported_claim`

**Persona B — Feasibility Critic**
- Critique types: `infeasible_step` | `underestimated_complexity` | `missing_dependency` | `resource_constraint_unaddressed` | `timeline_unrealistic`

**Persona C — Coverage Critic**
- Critique types: `gap_not_fully_closed` | `success_criteria_insufficient` | `sub_gap_missed` | `scope_mismatch`

**Output Schema — SolutionCriticOutput:**
```
field: run_id                     type: string
field: approach_id                type: string
field: gap_id                     type: string
field: critic_persona             type: enum: "methodological" | "feasibility" | "coverage"
field: iteration_number           type: integer
field: critiques                  type: SolutionCritique[]
field: overall_approach_assessment type: enum: "strong" | "needs_refinement" | "fundamental_flaw"
field: assessment_rationale       type: string  constraint: 50–150 words
field: confidence                 type: float   range: 0.0–1.0

SolutionCritique:
  field: critique_id              type: string  format: SC-{persona_initial}-{number}
  field: critique_type            type: enum    values: as listed per persona above
  field: claim                    type: string
  field: anchor                   type: EvidenceAnchor   constraint: REQUIRED
  field: severity                 type: enum: "fatal" | "major" | "minor"
  field: affects_step_id          type: string | null
  field: suggested_correction     type: string
  field: affects_gap_closure      type: boolean
```

---

### 5.11 Solution Analyzer Agent

**Role:** Receives all three Critic persona outputs for a given approach. Applies the rubric. Filters noise. Identifies consensus critiques. Decides: iterate or escalate to Solution Judge.

**Model:** Any capable model

**Rubric — Same 4 criteria as Gap Analyzer, re-interpreted for solution context:**
```
Criterion 1 — Anchored
  Pass: critique cites verbatim text from Literature MD, Approach MD, or Generator's own output

Criterion 2 — Specific
  Pass: critique points to a specific implementation step, claim, or evidence anchor

Criterion 3 — Actionable
  Pass: suggested_correction describes a specific change to a specific step or claim

Criterion 4 — Non-Redundant
  Pass: this critique raises a distinct issue not already covered by another valid critique

Validity threshold: total_passed >= 3 → is_valid = true
```

**Output Schema — SolutionAnalyzerOutput:** Same structure as GapAnalyzerOutput with field names adapted to solution context.

---

### 5.12 Solution LLM-as-Judge

**Role:** Evaluates the full solution iteration history. Uses adaptive web search to verify approach novelty and check for known failure modes. Makes final approval decision.

**Model:** Strongest available model (same as Gap Judge)

**Web Search MCP — Adaptive Search Loop:**
```
Round 1 — Broad orientation:
  Query types: "check_approach_precedent", "check_known_failures"

Round 2 — Targeted follow-up (if Round 1 confidence = "low"):
  Query types: "validate_citation", "follow_up_on_finding", "check_approach_precedent"
  Rule: must reference specific finding from Round 1

Round 3 — Single verification only (rare):
  1 query max. Always stop after.
```

**Quality Assessment Rubric:**
```
gap_coverage_score (0–10):
  10 = approach provably closes the gap; success criteria directly measure closure
  0  = approach does not close the gap; reject

methodological_rigor_score (0–10):
  10 = every step logically sequenced, fully supported by literature
  0  = approach is methodologically unsound; reject

feasibility_score (0–10):
  10 = all steps implementable with clearly available resources
  0  = not implementable as stated; reject

novelty_score (0–10):
  10 = approach is original; no direct precedent found
  0  = approach is a restatement of existing work; reassess

overall_score = (gap_coverage_score × 0.40) + (methodological_rigor_score × 0.30) + (feasibility_score × 0.20) + (novelty_score × 0.10)
```

**Decision Rules:**
```
overall_score >= 7.0 AND strategic_iteration_count < 2 → APPROVE
overall_score >= 5.0 AND strategic_iteration_count < 2 → CONTINUE
overall_score < 5.0 AND strategic_iteration_count >= 2 → REJECT
overall_score < 5.0 AND strategic_iteration_count < 2 → CONTINUE
```

**Output Schema — SolutionJudgeOutput:** Same structure as GapJudgeOutput with `search_session: AdaptiveSearchState` and quality assessment fields replaced by the four scores above.

---

### 5.13 Instruction Merger (Solution Loop)

**Role and rules:** Identical to Gap Instruction Merger (Section 5.6). Applied to Solution Generator instead of Gap Finder.

---

## 6. Control Flow and Feedback Rules

### 6.1 Tactical Loop (Analyzer → Loop Entry Point)

**Trigger:** Analyzer returns `decision = "iterate"`

**Steps:**
1. Analyzer produces `iteration_instruction`
2. Check if `judge_supersede_active = true` → if yes, skip Analyzer instruction, use only Judge's strategic instruction
3. Check if Judge has active strategic instruction with `supersedes_tactical = false` → if yes, go to Instruction Merger
4. Increment `tactical_iteration_count`
5. Deliver instruction to Generator
6. Generator produces new output → re-enter Critic → Analyzer

### 6.2 Strategic Loop (Judge → Loop Entry Point)

**Trigger:** Judge returns `decision = "CONTINUE"`

**Steps:**
1. Judge produces `strategic_instruction` and sets `supersedes_tactical`
2. If `supersedes_tactical = true`:
   - Set `judge_supersede_active = true` in IterationState
   - Clear any active Analyzer `iteration_instruction`
   - Increment `strategic_iteration_count`
   - Deliver only Judge's instruction to Generator
3. If `supersedes_tactical = false`:
   - If Analyzer has active instruction → go to Instruction Merger
   - If no active Analyzer instruction → deliver only Judge's instruction
   - Increment `strategic_iteration_count`
4. Generator produces new output → re-enter full Critic → Analyzer → Judge chain

### 6.3 Sequential Authority Protocol

```
RULE 1 — Judge supersedes by default when in conflict
  Direction change → supersedes_tactical = true
  Additive instruction → supersedes_tactical = false

RULE 2 — Analyzer resets when Judge supersedes
  When supersedes_tactical = true:
    Clear Analyzer's active iteration_instruction
    Set judge_supersede_active = true
    Generator receives only strategic_instruction

RULE 3 — Analyzer resumes after strategic instruction is addressed
  After Generator produces output under Judge's strategic instruction:
    Set judge_supersede_active = false
    Analyzer runs fresh on new output (previous tactical instructions do not carry over)

RULE 4 — Merger is the only path when both are active
  If supersedes_tactical = false AND Analyzer has active instruction:
    Route through Instruction Merger
    Generator receives only merged_instruction
    Never deliver raw Analyzer and Judge instructions separately in the same pass
```

### 6.4 Iteration Limits (Hard Limits — Enforced by Orchestrator)

```
MAX_TACTICAL_ITERATIONS     = 3
MAX_STRATEGIC_ITERATIONS    = 2
MAX_TOTAL_ITERATIONS        = 5
```

**Hard limit enforcement:**
```
IF total_iteration_count >= MAX_TOTAL_ITERATIONS:
  Force escalation to Judge with force_decision = true
  Judge MUST issue APPROVE or REJECT (CONTINUE not available)
  If REJECT: mark as "unresolvable", log reason, skip to next

Unresolvable gaps are not passed to the solution loop.
Unresolvable solutions noted in final output with last-best-effort approach attached.
```

---

## 7. Web Search MCP Integration

### 7.1 When It Is Used
Web Search MCP is available **only to the two Judge agents**. Not available to Critic, Analyzer, Generator, or Merger agents.

### 7.2 Adaptive Search Loop — How It Works

```
SEARCH LOOP FLOW:

Judge invoked
  → Formulate Round 1 queries (broad)
  → Execute queries via Web Search MCP
  → Synthesise round_findings_summary
  → Assess confidence_to_score
      ├── "high"   → STOP SEARCH → proceed to scoring
      ├── "medium" → STOP if round >= 2, else CONTINUE
      └── "low"    → CONTINUE if round < MAX_ROUNDS AND queries_remaining > 0
                         → Formulate Round 2 queries (targeted, based on Round 1)
                         → Execute → Assess
                             ├── "high" or "medium" → STOP SEARCH
                             └── "low" → CONTINUE to Round 3 (final, 1 query max)
                                           → Execute → STOP (always)
  → Compile AdaptiveSearchState from all rounds
  → Apply findings to quality score adjustments
  → Issue decision
```

### 7.3 Round-by-Round Query Strategy

| Round | Purpose | Query specificity | Max queries | Stop if |
|---|---|---|---|---|
| 1 | Broad orientation | Domain-level, general terms | 2 | Confidence = "high" |
| 2 | Targeted follow-up | Named entities from Round 1 | 2 | Confidence = "high" or "medium" |
| 3 | Single verification | One specific factual question | 1 | Always (final round) |

Round 2 queries must reference a specific output from Round 1. If Round 1 found nothing relevant, do not run Round 2.

### 7.4 When to Skip Search Entirely
- `iteration_count >= 3`
- Previous Judge for this gap/approach already ran search with `search_terminated_by = "no_relevant_results"`
- Domain is highly specialised and general web search will not return peer-reviewed content

When skipping: `search_terminated_by = "skipped"`, `web_search_informed = false`.

### 7.5 Impact on Quality Scores

```
Gap is definitively addressed in recent literature
  → novelty_score: reduce by 3–5 points minimum

Gap is partially addressed; key aspects remain open
  → novelty_score: reduce by 1–2 points

Citation misrepresented or does not support the claim
  → evidence_quality_score: reduce by 2–3 points per misrepresented citation

Proposed approach has documented failure modes
  → feasibility_score: reduce by 1–3 points

Proposed approach is original and supported
  → novelty_score: may increase by 1–2 points (cap at 10)

No relevant results found
  → No adjustment; note in score_rationale that novelty could not be independently verified
```

**Hard rule:** Do not APPROVE where web search returns clear, direct evidence of prior resolution.

### 7.6 Search Result Quality Handling
- Set `result_relevance` on each WebSearchRecord before round's `context_assessment`
- If all queries in a round return `not_relevant`: do not continue to next round
- Low-authority sources weighted lower in score adjustments
- Paywall pages with no abstract: `result_relevance = "partially_relevant"`

### 7.7 Configuration Variables for Search
```
MAX_SEARCH_ROUNDS                    default: 3
MAX_QUERIES_PER_ROUND                default: 2
MAX_TOTAL_QUERIES_PER_JUDGE_CALL     default: 5
SKIP_SEARCH_AT_ITERATION             default: 3
```

---

## 8. Final Output Schema

**ValidatedOutput** (one per approved gap–solution pair):
```
field: output_id                  type: string  format: OUT-{gap_id}
field: gap
  field: gap_id                   type: string
  field: title                    type: string
  field: description              type: string
  field: gap_type                 type: enum
  field: severity                 type: enum
  field: evidence                 type: GapEvidence[]
  field: novelty_score            type: float
  field: evidence_quality_score   type: float
  field: gap_clarity_score        type: float
  field: gap_iteration_count      type: integer

field: solution
  field: approach_id              type: string
  field: approach_title           type: string
  field: approach_summary         type: string
  field: gap_closure_statement    type: string
  field: implementation_steps     type: ImplementationStep[]
  field: known_limitations        type: string[]
  field: success_criteria         type: string[]
  field: literature_alignment     type: LiteratureAlignment[]
  field: gap_coverage_score       type: float
  field: methodological_rigor_score type: float
  field: feasibility_score        type: float
  field: solution_iteration_count type: integer

field: audit_trail
  field: gap_discovery_iterations type: integer
  field: solution_iterations      type: integer
  field: web_search_queries_used  type: integer
  field: key_web_findings         type: string[]
  field: final_gap_judge_rationale type: string
  field: final_solution_judge_rationale type: string
```

---

## 9. Implementation Sequence

Build and validate in this order to avoid debugging compound failures:

**Stage 1 — Scaffold**
1. Implement Orchestrator state machine (no LLM calls — just routing logic and iteration counters)
2. Implement hard limit enforcement and status transitions
3. Validate state schema with mock data

**Stage 2 — Gap Finder loop (no review)**
4. Implement Gap Finder agent with output schema validation
5. Test Gap Finder on sample Literature MD + Approach MD (Refinement Mode)
6. Test Gap Finder on sample Literature MD only (Discovery Mode)
7. Validate GapEvidence anchor presence and mode-aware validation rules

**Stage 3 — Gap Review chain**
8. Implement Gap Critic (all 3 personas as separate prompt runs)
9. Implement Gap Analyzer with rubric scoring
10. Connect Gap Finder → Critic → Analyzer → back to Finder
11. Test tactical loop with mock data, verify iteration counter increments
12. Validate that unanchored critiques are correctly rejected by rubric

**Stage 4 — Gap Judge**
13. Implement Gap Judge without web search first
14. Test approve/continue/reject paths
15. Validate sequential authority: test `supersedes_tactical = true` path
16. Add web search MCP integration
17. Test adaptive search loop: verify early stop on high confidence

**Stage 5 — Instruction Merger (Gap)**
18. Implement Instruction Merger
19. Test with deliberately conflicting tactical + strategic instructions
20. Verify only one merged instruction reaches Gap Finder

**Stage 6 — Gap Prioritization**
21. Implement prioritization step
22. Test ranking with 5+ approved gaps

**Stage 7 — Base Paper Selection**
23. Implement candidate shortlisting (Step 1) with all 5 scoring criteria
24. Implement human decision gate — Orchestrator pauses, presents shortlist, waits for researcher input
25. Implement Base Paper Profile generation (Step 3)
26. Test: verify pipeline pauses correctly and resumes after researcher selection
27. Test: verify researcher-specified paper (not in shortlist) path works
28. Test: verify checkpoint saves state during human gate pause

**Stage 8 — Solution loop**
29. Implement Solution Generator
30. Implement Solution Critic (3 personas)
31. Implement Solution Analyzer
32. Implement Solution Judge with web search
33. Implement Instruction Merger (Solution)
34. Connect full solution chain

**Stage 9 — End-to-end**
35. Run full pipeline in Discovery Mode (no Approach MD) on real Literature MD
36. Run full pipeline in Refinement Mode (with Approach MD)
37. Verify human gate pauses correctly at Base Paper Selection in both modes
38. Inspect audit trail for each output
39. Check web search queries are well-formed and findings influence scores correctly

---

## 10. Key Configuration Variables

```
# Pipeline iteration limits
MAX_TACTICAL_ITERATIONS           default: 3
MAX_STRATEGIC_ITERATIONS          default: 2
MAX_TOTAL_ITERATIONS              default: 5
TOP_N_GAPS_FOR_SOLUTION_LOOP      default: 5

# Scoring thresholds
GAP_APPROVAL_SCORE_THRESHOLD      default: 7.0
GAP_CONTINUE_SCORE_THRESHOLD      default: 5.0
SOLUTION_APPROVAL_SCORE_THRESHOLD default: 7.0
SOLUTION_CONTINUE_SCORE_THRESHOLD default: 5.0

# Web search
MAX_SEARCH_ROUNDS                 default: 3
MAX_QUERIES_PER_ROUND             default: 2
MAX_TOTAL_QUERIES_PER_JUDGE_CALL  default: 5
SKIP_SEARCH_AT_ITERATION          default: 3

# Model assignments
GENERATOR_MODEL                   required
CRITIC_MODEL                      required: must differ from GENERATOR_MODEL
JUDGE_MODEL                       required
ANALYZER_MODEL                    default: same as GENERATOR_MODEL
REASONING_LEVEL                   default: "high"  applies to all Nemotron and GPT-OSS adapters
                                  override per-role via ModelConfig if latency is a concern
                                  Analyzer and Merger are the best candidates for "low" override

# Execution
EXECUTION_MODE                    default: "sequential"  values: "sequential" | "parallel"

# GPU
GPU_MODE                          default: "auto"  values: "auto" | "manual" | "single"
GPU_VRAM_HEADROOM_GB              default: 8

# Checkpoint
CHECKPOINT_ENABLED                default: true
CHECKPOINT_GRANULARITY            default: "iteration"  values: "agent" | "iteration" | "gap"
CHECKPOINT_STORAGE                default: "filesystem"  values: "filesystem" | "sqlite"
CHECKPOINT_DIR                    default: "{project_root}/.anvesha/checkpoints"
MAX_AGENT_RETRIES                 default: 3
RETRY_BACKOFF_SECONDS             default: 5
```


---

## 11. SDK Package Structure

### 11.1 Context

**Anvesha** is the Research Assistant Kit — a multi-phase SDK that automates the research workflow end-to-end. Each phase corresponds to a stage in a research project and maps to a markdown file in the research project workspace.

This implementation plan covers **Phase 3: Research Gaps** — the third phase of Anvesha. Phase 2 (Literature Survey) produces the Literature MD that Phase 3 consumes. Phase 3 produces the Research Gaps MD that later phases build on. (Phase 1, Filter Literature, produces the curated reading list that Phase 2 surveys.)

There are three distinct structures, each with a different owner and purpose:

| Structure | Location | Owner | Purpose |
|---|---|---|---|
| Anvesha SDK | `anvesha/` Python package (PyPI) | Anvesha team | The code that runs; installed via pip; never edited by the researcher |
| Research Project Workspace | `research_project/` on researcher's machine | Researcher | Phase markdown outputs, versioning, config; what Anvesha reads and writes |
| Researcher's Code | Separate git repo — lives wherever the researcher keeps it | Researcher | Experiment code, training scripts, model configs; fully researcher-owned |

The researcher's code (Phase 8 training, evaluation scripts, etc.) is a **completely separate git repository** that Anvesha does not own or manage. Anvesha only needs to know its path via `.anvesha/config.yaml` to trigger runs from it on the GPU box.

---

### 11.2 Research Project Workspace

The directory structure on the researcher's machine. Anvesha reads from and writes to this structure. It is not part of the Python package — the researcher owns it.

```
research_project/                        # researcher-owned root directory
│
├── README.md                            # project overview (human-maintained)
├── _VERSION_LOG.md                      # Anvesha writes here after each phase completes
│
├── .anvesha/                            # hidden — Anvesha-managed, gitignored
│   ├── config.yaml                      # project-level Anvesha config (models, GPU, paths)
│   │                                    # includes: researcher's code repo path (for Phase 8)
│   └── checkpoints/                     # phase run checkpoints for resume
│       ├── phase03_run_abc123/
│       └── ...
│
├── phases/                              # Anvesha reads and writes here
│   ├── 01_filtered_literature.md        # Phase 1 output (curated reading list)
│   ├── 02_literature_survey.md          # Phase 2 output → INPUT to Phase 3 (Literature MD)
│   ├── 03_approach.md                   # OPTIONAL — researcher's current approach
│   │                                    # if present: Gap Finder runs in Refinement Mode
│   │                                    # if absent:  Gap Finder runs in Discovery Mode
│   │                                    # researcher creates this manually if needed
│   ├── 03_research_gaps.md              # Phase 3 output ← this pipeline writes here
│   ├── 04_base_paper.md                 # Phase 4 output ← Base Paper Profile
│   ├── 05_ideas.md                      # Phase 5 (future)
│   ├── 06_experiment_plan.md            # Phase 6 (future)
│   ├── 07_implementation_notes.md       # Phase 7 (future)
│   ├── 08_results.txt                   # Phase 8 — Anvesha pulls this from GPU box
│   ├── 09_evaluation.md                 # Phase 9 — Evaluation & Analysis (merged)
│   ├── 10_future_ideation.md            # Phase 10 (future)
│   ├── 11_loop_decision.md              # Phase 11 (future)
│   │
│   └── (versioned re-runs — Anvesha creates these automatically)
│       ├── 05_ideas_v2.md
│       ├── 06_experiment_plan_v2.md
│       └── ...
│
├── data/                                # RESEARCHER-OWNED — local datasets, gitignored
│
├── refs/                                # RESEARCHER-OWNED — paper PDFs and citations
│   ├── pdfs/
│   └── bibtex.bib
│
└── runs/                                # Anvesha pulls logs here from GPU box (Phase 8)
    ├── run_001/                         # researcher also browses these manually
    └── run_002/
```

**Phase I/O mapping (Phases 2–4, 8 — those that interact with Phase 3):**
```
Phase 2  reads:  phases/01_filtered_literature.md  → curated reading list
         writes: phases/02_literature_survey.md    → synthesised Literature MD

Phase 3  reads:  phases/02_literature_survey.md    → Literature MD
                 phases/03_research_gaps.md         → prior gap notes (if re-run)
                 phases/03_approach.md              → researcher's current approach (optional)
                                                      if present: Refinement Mode
                                                      if absent:  Discovery Mode
         writes: phases/03_research_gaps.md         → validated gaps + approaches
                 _VERSION_LOG.md                    → updated with run ID and timestamp

Phase 4  reads:  phases/03_research_gaps.md         → prioritised gap list
                 phases/02_literature_survey.md     → for candidate paper search
         pauses: human decision gate (researcher selects base paper)
         writes: phases/04_base_paper.md            → Base Paper Profile
                 _VERSION_LOG.md

Phase 8  reads:  researcher's code repo path from .anvesha/config.yaml
                 (code repo lives outside research_project/ — separate git repo)
                 Anvesha does not read or modify code repo files
         triggers: training job on GPU box via SSH
         pulls:  training logs → runs/run_00N/
         writes: phases/08_results.txt              → summary of training outcomes
```

---

### 11.3 Anvesha SDK Package Structure

The Python package. Shared infrastructure (`core/`) is designed to be reused across all phases. Phase-specific logic lives under `phases/phase03_gaps/` and does not bleed into other phases.

```
anvesha/
├── __init__.py                          # public API surface
├── cli.py                               # anvesha CLI entry point
│                                        # commands: init | run | resume | status | logs
│
├── core/                                # shared infrastructure across all phases
│   │
│   ├── adapters/                        # LLM provider adapters
│   │   ├── base.py                      # ModelAdapter protocol
│   │   │
│   │   ├── local/                       # primary — local-first
│   │   │   ├── __init__.py
│   │   │   ├── vllm.py                  # vLLM server
│   │   │   │                            # — harmony_format flag (required for GPT-OSS-120B)
│   │   │   │                            # — reasoning_effort: default "high"
│   │   │   │                            # — guided_json structured output support
│   │   │   ├── nemotron.py              # Nemotron 3 Super
│   │   │   │                            # — reasoning_level: default "high" for all roles
│   │   │   │                            # — Mamba-Transformer chat template injection
│   │   │   │                            # — TensorRT-LLM or vLLM backend selectable
│   │   │   │                            # — 1M token context window support
│   │   │   ├── ollama.py                # Ollama + LM Studio (OpenAI-compat)
│   │   │   ├── llamacpp.py              # llama.cpp server mode (OpenAI-compat)
│   │   │   └── huggingface.py           # in-process HF transformers (no server)
│   │   │
│   │   ├── cloud/                       # provision/fallback only
│   │   │   ├── __init__.py
│   │   │   ├── anthropic.py
│   │   │   ├── openai.py
│   │   │   └── google.py
│   │   │
│   │   └── fallback.py                  # FallbackAdapter
│   │                                    # — wraps local primary + cloud backup
│   │                                    # — triggers: timeout | parse_error | gpu_oom | low_confidence
│   │
│   ├── gpu/                             # GPU node management
│   │   ├── manager.py                   # GpuManager — central node state
│   │   ├── scheduler.py                 # sequential load/run/unload cycle
│   │   ├── allocator.py                 # model-to-node assignment (auto or manual)
│   │   └── probe.py                     # CUDA device detection and VRAM reporting
│   │
│   ├── checkpoint/                      # pipeline state persistence and resume
│   │   ├── manager.py                   # CheckpointManager — save and load
│   │   ├── serializer.py                # PipelineState ↔ JSON or SQLite
│   │   ├── resume.py                    # ResumePlanner — last safe restart point
│   │   └── recovery.py                  # FailureRecovery — per-failure-type rules
│   │
│   ├── config/                          # configuration classes
│   │   ├── pipeline_config.py           # PipelineConfig — assembles all sub-configs
│   │   ├── model_config.py              # ModelConfig — adapter per role, per-role overrides
│   │   ├── gpu_config.py                # GpuConfig — node IDs, mode, VRAM headroom
│   │   ├── search_config.py             # SearchConfig — rounds, query limits, skip rules
│   │   └── rubric_config.py             # RubricConfig — score thresholds, iteration limits
│   │
│   └── hooks.py                         # PipelineHooks — pre/post agent callbacks
│
├── workspace/                           # research project workspace management
│   ├── project.py                       # ResearchProject — locates and validates project root
│   │                                    # reads project metadata, resolves phase file paths
│   ├── phase_io.py                      # reads and writes phase markdown files
│   │                                    # handles versioning (v2, v3 suffixes)
│   ├── version_log.py                   # _VERSION_LOG.md read/write
│   └── schemas.py                       # WorkspaceConfig, PhaseFilePaths, ProjectMeta
│
└── phases/
    ├── __init__.py
    │
    ├── phase01_filter/                  # Phase 1: Filter Literature (future)
    │   └── (placeholder)
    │
    ├── phase02_survey/                  # Phase 2: Literature Survey (future)
    │   └── (placeholder)
    │
    ├── phase03_gaps/                    # Phase 3: Research Gap Analysis ← THIS PIPELINE
    │   ├── __init__.py
    │   ├── pipeline.py                  # ResearchGapPipeline — Phase 3 entry point
    │   │                                # reads workspace inputs, runs pipeline,
    │   │                                # writes 03_research_gaps.md on completion
    │   ├── orchestrator.py              # state machine, routing, iteration counters
    │   ├── state.py                     # PipelineState, GapReviewState, SolutionReviewState
    │   │
    │   ├── schemas/
    │   │   ├── shared.py                # EvidenceAnchor, IterationState, RubricScore
    │   │   │                            # SearchRound, AdaptiveSearchState, WebSearchRecord
    │   │   ├── gap.py                   # GapFinderOutput, GapCriticOutput
    │   │   │                            # GapAnalyzerOutput, GapJudgeOutput
    │   │   └── solution.py              # SolutionGeneratorOutput, SolutionCriticOutput
    │   │                                # SolutionAnalyzerOutput, SolutionJudgeOutput
    │   │
    │   ├── agents/
    │   │   ├── base.py                  # BaseAgent, BaseCritic, BaseAnalyzer, BaseJudge
    │   │   ├── gap_finder.py
    │   │   ├── gap_critic.py            # 3-persona runner
    │   │   ├── gap_analyzer.py          # rubric scoring engine
    │   │   ├── gap_judge.py             # adaptive search loop + scoring
    │   │   ├── solution_generator.py
    │   │   ├── solution_critic.py       # 3-persona runner
    │   │   ├── solution_analyzer.py
    │   │   ├── solution_judge.py
    │   │   ├── prioritizer.py           # gap prioritization between loops
    │   │   └── merger.py                # instruction merger
    │   │
    │   ├── loops/
    │   │   ├── gap_loop.py              # GapDiscoveryLoop
    │   │   └── solution_loop.py         # SolutionLoop
    │   │
    │   └── search/
    │       └── web_search.py            # adaptive search loop, MCP client wrapper
    │
    ├── phase04_base_paper/              # Phase 4: Base Paper (future)
    │   └── (placeholder)
    │
    ├── phase05_ideas/                   # Phase 5: Idea Generation (future)
    │   └── (placeholder)
    │
    ├── phase06_experiment/              # Phase 6: Experiment Plan (future)
    │   └── (placeholder)
    │
    ├── phase07_implementation/          # Phase 7: Implementation Notes (future)
    │   └── (placeholder)
    │
    ├── phase08_training/                # Phase 8: Training — thin remote runner
    │   ├── __init__.py
    │   ├── pipeline.py                  # TrainingPhase — Phase 8 entry point
    │   │                                # reads .anvesha/config.yaml for GPU box SSH config
    │   │                                # and researcher's code repo path
    │   │                                # does NOT touch code repo files directly
    │   ├── remote.py                    # SSH connection to GPU box
    │   │                                # submits training job (sbatch, python, or custom script)
    │   │                                # streams stdout/stderr into runs/run_00N/
    │   ├── log_puller.py                # pulls training logs from GPU box to runs/
    │   │                                # runs while job is active
    │   └── results_writer.py            # summarises training outcome into phases/08_results.txt
    │
    ├── phase09_evaluation/              # Phase 9: Evaluation & Analysis (merged) — future
    │   └── (placeholder)
    │
    ├── phase10_future_ideation/         # Phase 10: Future Ideation (future)
    │   └── (placeholder)
    │
    └── phase11_loop_gate/               # Phase 11: Loop Decision Gate (future)
        └── (placeholder)
```

---

### 11.4 Boundary Rules

```
RULE 1 — Phase modules do not import from other phase modules
  phases/phase03_gaps/ must not import from phases/phase02_survey/
  Phase outputs travel via workspace markdown files, not in-memory objects
  This keeps every phase independently runnable

RULE 2 — Core is phase-agnostic
  core/ contains no phase-specific logic
  Adapters, GPU, checkpoint, and config are identical across all phases

RULE 3 — Workspace module owns all file I/O
  No agent, loop, or phase module reads or writes files directly
  All file access goes through workspace/phase_io.py
  Agents receive strings (MD content), not file paths

RULE 4 — Anvesha never writes into the researcher's code repo
  The code repo is researcher-owned and managed by their own git workflow
  Phase 8 (Training) reads the repo path from .anvesha/config.yaml to locate it
  Phase 8 triggers runs on the GPU box but does not modify code repo files

RULE 5 — .anvesha/ is always gitignored in the research project repo
  Checkpoints and run state are machine-local, not version-controlled
  The research_project/ .gitignore must include .anvesha/

RULE 6 — Phase entry points accept ResearchProject, not raw file paths
  Each phase pipeline.py accepts a ResearchProject object
  ResearchProject (from workspace/project.py) resolves all file paths
  Phases are never called with hardcoded paths

RULE 7 — Phase 8 (Training) is always a thin wrapper, never a trainer
  Training code lives in the researcher's code repo, not in Anvesha
  phase08_training/ only handles: SSH, job submission, log pulling, result writing
  If training logic ever appears in Anvesha, it belongs in the researcher's code repo
```

---

### 11.5 PyPI Packaging and CLI Usage

Anvesha is distributed as a single PyPI package. All phases are included. Researchers install once and run by phase number.

```
pip install anvesha

# initialise a new research project workspace
anvesha init --name "my-research-project"
# creates research_project/ with all directories and .anvesha/config.yaml

# run a phase
anvesha run --phase 1
anvesha run --phase 3
anvesha run --phase 8 --gpu-box user@gpu-server.university.edu

# resume a phase after failure or crash
anvesha resume --phase 3

# check status of all phases for current project
anvesha status

# view logs of a completed or running phase
anvesha logs --phase 3 --follow
```

Each `anvesha run --phase N` call:
1. Validates the project root (looks for `.anvesha/config.yaml`)
2. Runs the phase pipeline in the foreground of the current terminal
3. On completion: writes the phase output MD and updates `_VERSION_LOG.md`

Anvesha has no knowledge of tmux. It is a foreground process. The researcher is responsible for running it inside tmux if they need connection resilience.

---

### 11.6 tmux — Researcher's Responsibility

Anvesha is tmux-agnostic. It runs as a normal foreground process. For long-running phases (Phase 3, Research Gaps, especially), the researcher manages tmux themselves.

```
RECOMMENDED WORKFLOW FOR LONG PHASES:

# on the server (local or remote):
tmux new -s phase3
anvesha run --phase 3

# detach safely at any time:
Ctrl+B  D

# reconnect later (same session, process still running):
tmux attach -t phase3

# if the process crashed (OOM, timeout, etc.):
anvesha resume --phase 3
# loads last checkpoint, restarts from that iteration
```

Anvesha does not check for tmux, depend on it, or manage sessions. If a connection drops without tmux, the process dies. The checkpoint system handles recovery — `anvesha resume` picks up from the last saved iteration. No work beyond the current iteration is lost.

---

### 11.7 Config Variable Update

```
CHECKPOINT_DIR    default: "{project_root}/.anvesha/checkpoints"
```

---

## 12. Execution Model — Sequential by Default

### 12.1 Why Sequential

Sequential execution is the default because it allows two large local models (Generator and Critic) to share the same GPU nodes without requiring both to be resident in VRAM simultaneously. Since the Critic only runs after the Generator has finished, the Generator can be unloaded before the Critic loads. This halves the peak VRAM requirement compared to parallel execution.

### 12.2 Sequential Execution Flow

```
SEQUENTIAL AGENT CALL CYCLE:

1. GPU Scheduler receives call request for Agent X
2. Check which model is currently loaded on the target node(s)
   IF the correct model is already loaded → skip to step 5
   IF a different model is loaded → proceed to step 3
3. Unload current model from node(s)
   — release VRAM, confirm free memory
   — log: "unloaded [model] from node [id], freed [N]GB"
4. Load Agent X's model onto node(s)
   — wait for model to be ready (health check)
   — log: "loaded [model] on node [id], using [N]GB"
5. Execute agent call → receive output
6. Return output to caller
   — do NOT unload yet (next call may use the same model)
   — GPU Scheduler caches which model is currently warm on which nodes
```

**Warm model caching:** If consecutive calls use the same model (e.g., all 3 Critic personas), the model stays loaded. Unloading only happens when a different model is requested.

### 12.3 Execution Mode Configuration

```
execution_mode: "sequential"  (default)
  — one model loaded at a time per GPU node
  — GPU Scheduler manages load/unload between agent calls
  — lower VRAM requirement, higher latency per pipeline run

execution_mode: "parallel"
  — multiple models may be loaded simultaneously
  — GPU Allocator assigns distinct nodes to each model at startup
  — higher VRAM requirement, lower latency
  — only viable if total VRAM across nodes covers all models simultaneously
```

### 12.4 Recommended Sequential GPU Layout

```
Nemotron 3 Super — Generator / Judge
  VRAM required: ~64GB
  Node assignment: [node_0, node_1]  (2×H100 80GB)

GPT-OSS-120B — Critic
  VRAM required: ~40GB
  Node assignment: [node_0]  (shared with Generator sequentially)

Sequential schedule for one gap iteration:
  Step 1: Load Nemotron 3 Super on [0,1] → run Gap Finder → keep warm
  Step 2: Unload Nemotron from [0,1]
  Step 3: Load GPT-OSS-120B on [0] → run Gap Critic (all 3 personas) → keep warm
  Step 4: Unload GPT-OSS-120B from [0]
  Step 5: Load Analyzer model on [0] → run Gap Analyzer
  Step 6: Load Nemotron on [0,1] → run Gap Judge
  Repeat for next gap
```

---

## 13. GPU Node Configuration

### 13.1 GPU Config Schema

```
GpuConfig:
  field: mode                   type: enum    values: "auto" | "manual" | "single"
                                default: "auto"

  field: execution_mode         type: enum    values: "sequential" | "parallel"
                                default: "sequential"

  field: available_node_range   type: tuple[int, int]   default: (0, 7)
                                description: min and max CUDA device IDs to consider

  field: vram_headroom_gb       type: float   default: 8.0

  field: manual_assignments     type: ModelNodeAssignment | null

  field: single_node_id         type: int | null   default: 0

ModelNodeAssignment:
  field: generator_nodes        type: list[int]
  field: critic_nodes           type: list[int]
  field: analyzer_nodes         type: list[int]
  field: judge_nodes            type: list[int]
  field: merger_nodes           type: list[int]
```

### 13.2 Auto-Detection Logic (mode = "auto")

```
AUTO GPU DETECTION SEQUENCE:

1. Enumerate all CUDA devices in available_node_range
   For each device: record device_id, total_vram_gb, free_vram_gb, device_name

2. Filter to usable devices:
   usable = devices where free_vram_gb > vram_headroom_gb

3. Determine VRAM requirement per model role from adapter.reported_vram_gb

4. In sequential mode:
   — find node group with largest total free VRAM
   — assign all roles to that group (they time-share)
   — verify largest_role_vram_needed <= total_free_vram_of_group
   — if not: raise InsufficientVramError with specific numbers

5. In parallel mode:
   — greedily assign each role to smallest available node group meeting VRAM requirement
   — if not enough: raise InsufficientVramError, suggest switching to sequential

6. Log final assignment summary before first agent call
```

### 13.3 Manual Mode Example

```
GpuConfig:
  mode: "manual"
  execution_mode: "sequential"
  manual_assignments:
    generator_nodes: [0, 1]
    critic_nodes:    [0, 1]    # shares nodes 0,1 sequentially with Generator
    analyzer_nodes:  [0]
    judge_nodes:     [0, 1]
    merger_nodes:    [0]
```

### 13.4 Single Mode Example

```
GpuConfig:
  mode: "single"
  execution_mode: "sequential"
  single_node_id: 0
  vram_headroom_gb: 8.0
```

### 13.5 VRAM Requirements Reference

```
VLLMAdapter (GPT-OSS-120B):
  reported_vram_gb: 40
  recommended_nodes: 1×H100 80GB

NemotronAdapter (Nemotron 3 Super):
  reported_vram_gb: 64
  recommended_nodes: 2×H100 80GB or 1×H200 or 1×B200

OllamaAdapter:
  reported_vram_gb: user-declared (no automatic detection via Ollama API)

HuggingFaceAdapter:
  reported_vram_gb: auto-detected at model load time
```

### 13.6 GPU Health Check at Startup

1. Probe all assigned nodes — confirm CUDA availability
2. Confirm free VRAM meets largest single-role requirement
3. Run lightweight test inference on each assigned adapter
4. If any node fails: raise `GpuNodeUnavailableError` with node ID and reason
5. Log GPU readiness summary before first agent call

---

## 14. Checkpoint and Resume System

### 14.1 Purpose

Long pipeline runs can fail midway due to OOM errors, network timeouts, MCP failures, or process interruptions. The checkpoint system allows the pipeline to resume from the last successful step without reprocessing completed work.

### 14.2 Checkpoint Granularity Levels

```
"agent"      Save after every individual agent call (most granular)
             — resume from the exact agent call that failed
             — recommended for: unstable environments, large pipelines

"iteration"  Save after each complete iteration (default)
             — one save per Critic → Analyzer → Judge cycle
             — resumes from the start of the failed iteration

"gap"        Save after each gap is fully resolved (coarsest)
             — fastest during normal execution
             — on failure: re-runs the entire current gap from scratch
```

### 14.3 Checkpoint State Schema

```
CheckpointState:
  field: pipeline_run_id             type: string
  field: checkpoint_version          type: integer
  field: created_at                  type: ISO timestamp
  field: last_saved_at               type: ISO timestamp
  field: granularity                 type: enum: "agent" | "iteration" | "gap"
  field: pipeline_state              type: PipelineState
  field: resume_hint
    field: phase                     type: enum: "gap_discovery" | "prioritization" | "solution_generation"
    field: current_gap_id            type: string | null
    field: current_approach_id       type: string | null
    field: last_completed_step       type: string
    field: last_completed_at         type: ISO timestamp
  field: failure_log                 type: FailureRecord[]

FailureRecord:
  field: failed_at                   type: ISO timestamp
  field: gap_id                      type: string | null
  field: agent_name                  type: string
  field: failure_type                type: enum: "timeout" | "parse_error" | "gpu_oom" | "search_failure" | "unknown"
  field: error_message               type: string
  field: recovery_action             type: string
```

### 14.4 Resume Logic

```
RESUME DECISION SEQUENCE:

1. Scan CHECKPOINT_DIR for files matching: {pipeline_run_id}_*.checkpoint.json
   IF no files found → start fresh
   IF files found → load most recent by checkpoint_version

2. Validate checkpoint:
   — confirm pipeline_run_id matches
   — confirm input hashes (Literature MD + Approach MD) match stored hashes
   — if inputs changed → warn user, offer: resume anyway | start fresh

3. Reconstruct PipelineState from checkpoint

4. Identify gaps by status:
   "approved" or "rejected" → skip entirely
   "in_review"              → resume from resume_hint.last_completed_step
   "pending"                → process fresh

5. For the gap being resumed:
   — restore iteration_state from checkpoint
   — restart from the agent AFTER last_completed_step

6. Log resume summary before starting
```

### 14.5 Failure Recovery Rules

```
FAILURE TYPE: timeout
  Action 1: retry up to MAX_AGENT_RETRIES with exponential backoff
  Action 2: if all retries fail and FallbackAdapter configured → use fallback
  Action 3: if no fallback → mark gap as "failed", log, continue to next

FAILURE TYPE: parse_error
  Action 1: retry once with schema-reminder appended to prompt
  Action 2: retry with simplify_schema_for_retry=true (strips optional fields)
  Action 3: if still failing → mark gap as "failed", log raw output, continue

FAILURE TYPE: gpu_oom
  Action 1: attempt to unload all models from affected nodes
  Action 2: reduce context length by 20% and retry
  Action 3: if still OOM and FallbackAdapter configured → route to cloud
  Action 4: if no fallback → halt with GpuOomError

FAILURE TYPE: search_failure
  Action 1: retry once after 3 seconds
  Action 2: if retry fails → continue Judge call without search
             set web_search_informed = false, search_terminated_by = "skipped"
  NOTE: search failure never stops the pipeline

FAILURE TYPE: checkpoint_write_failure
  Action 1: log warning, continue execution (state still in memory)
  NOTE: pipeline never stops due to checkpoint failure alone
```

### 14.6 Checkpoint File Management

```
File naming:     {pipeline_run_id}_{checkpoint_version:05d}.checkpoint.json
Retention:       Keep last 5 checkpoint files per run_id
Storage:
  "filesystem"   JSON files in CHECKPOINT_DIR (default)
  "sqlite"       Single SQLite file — atomic writes, safer for "agent" granularity
On completion:   Final checkpoint written with status="completed"
```

### 14.7 Input Hash Validation

```
InputHashes:
  field: literature_md_hash    type: string   SHA-256 of literature MD content
  field: approach_md_hash      type: string | null   SHA-256 of approach MD content
                                                     null in Discovery Mode
  field: hashed_at             type: ISO timestamp
```

On resume, if hashes differ: warn user, offer resume-anyway or start-fresh. Default: start fresh.

---

## 15. Adapter Specifications — Local Models

### 15.1 VLLMAdapter (GPT-OSS-120B and general vLLM)

**Configuration schema:**
```
VLLMAdapterConfig:
  field: base_url              type: string   default: "http://localhost:8000/v1"
  field: model_name            type: string
  field: harmony_format        type: boolean  default: false
                               description: set true for GPT-OSS-120B and GPT-OSS-20B
                               when true: applies Harmony chat template instead of ChatML
                               Standard ChatML will cause GPT-OSS models to behave incorrectly
  field: reasoning_effort      type: enum | null  values: "low" | "medium" | "high"
                               default: "high"
                               description: applies to GPT-OSS models via Harmony Format only
                               injected as "Reasoning: {level}" in system prompt
                               per-role override via ModelConfig
  field: api_key               type: string   default: "EMPTY"
  field: timeout_seconds       type: integer  default: 120
  field: reported_vram_gb      type: float
```

**Structured output:** Uses vLLM's `guided_json` parameter with Pydantic model's JSON schema. Falls back to prompt-based schema instruction if not supported.

### 15.2 NemotronAdapter (Nemotron 3 Super)

**Configuration schema:**
```
NemotronAdapterConfig:
  field: base_url              type: string   default: "http://localhost:8000/v1"
  field: model_name            type: string   default: "nvidia/nemotron-3-super-120b-a12b"
  field: backend               type: enum     values: "tensorrt_llm" | "vllm"
                               default: "vllm"
  field: reasoning_level       type: enum     values: "low" | "medium" | "high"
                               default: "high"
                               description:
                                 Global default for all agent roles using this adapter.
                                 "high" is recommended for research pipelines without
                                 strict latency requirements.
                                 If speed becomes a bottleneck, drop Analyzer and Merger
                                 to "low" first — highest saving, negligible quality impact.
                                 Per-role override via ModelConfig.
  field: context_window        type: integer  default: 1_000_000
  field: timeout_seconds       type: integer  default: 240
  field: reported_vram_gb      type: float    default: 64.0
```

**Reasoning level injection:**
- Injected as header into system prompt: `"<reasoning_level>{level}</reasoning_level>\n{system_prompt}"`
- All roles default to "high"
- Per-role override: set `reasoning_level` in `ModelConfig` for that role

### 15.3 OllamaAdapter

**Configuration schema:**
```
OllamaAdapterConfig:
  field: base_url              type: string   default: "http://localhost:11434/v1"
  field: model_name            type: string
  field: api_key               type: string   default: "ollama"
  field: timeout_seconds       type: integer  default: 90
  field: reported_vram_gb      type: float    description: user-declared
  field: simplify_schemas      type: boolean  default: true
```

### 15.4 FallbackAdapter

**Configuration schema:**
```
FallbackAdapterConfig:
  field: primary               type: ModelAdapter   description: local adapter (tried first)
  field: fallback              type: ModelAdapter   description: cloud adapter
  field: fallback_on           type: list[enum]
                               values: "timeout" | "parse_error" | "gpu_oom" | "low_confidence" | "explicit"
                               default: ["timeout", "parse_error", "gpu_oom"]
  field: log_fallback_events   type: boolean  default: true
  field: fallback_timeout_seconds type: integer  default: 60
```

**Fallback behaviour:**
- Same input sent to fallback adapter unchanged when trigger condition met
- If fallback also fails: raise original error, do not swallow it
