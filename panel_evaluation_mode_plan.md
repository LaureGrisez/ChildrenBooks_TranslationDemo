# Panel Evaluation Mode Implementation Plan

## Goal

Add `EVALUATION_MODE=panel` without changing candidate generation or the
existing `single` evaluation behavior. Keep `WORKFLOW_MODE=text|multimodal`
independent from evaluation mode.

The first implementation should favor inspectability and reliable recovery over
an overly dynamic graph. `adaptive_panel` should follow after the fixed panel
has been calibrated.

## Architectural Direction

Keep the existing candidate-generation action and select an evaluation strategy
after it:

```text
generate_candidates
        |
        +-- EVALUATION_MODE=single
        |       critique_candidates
        |       summarize_critic
        |       generate_final_text
        |
        +-- EVALUATION_MODE=panel
                align_candidate_paragraphs
                run_panel_judges
                aggregate_panel_judgments
                generate_final_paragraphs
                audit_book_consistency
                repair_flagged_paragraphs
                finalize_translation
```

The `single` path should retain its current prompts, state keys, and report
format. The panel path should write the same final `final_translations` state
key so downstream saving and display code remains reusable.

Do not put all panel logic into `workflow.py`. Burr actions should orchestrate
small domain modules that can be tested without running Burr or calling models.

## Proposed Modules

### Existing modules to preserve

- `src/translation/workflow.py`: Burr graph construction and action
  orchestration.
- `src/translation/prompts.py`: existing single-mode prompts.
- `src/translation/reporting.py`: current single-mode report and common final
  artifact persistence.
- `src/translation/segmentation.py`: source paragraph splitting and spread
  mapping.

### New or extracted modules

- `src/translation/providers.py`
  - Shared provider request interface used by candidates, judges, synthesis,
    and audits.
  - Own OpenAI/Anthropic/Gemini calls, retry policy, cache integration, and
    optional image payloads.
  - Replaces the provider-specific call duplication currently in
    `workflow.py`.

- `src/translation/panel_models.py`
  - Dataclasses or typed structures for `JudgeSpec`, `ParagraphOption`,
    `EvaluationBlock`, `JudgeResult`, `AggregatedJudgment`, `AuditFinding`, and
    `RepairResult`.
  - Centralizes serialization contracts and validation.

- `src/translation/alignment.py`
  - Align source and candidate paragraphs.
  - Validate paragraph counts and preserve stable paragraph IDs.
  - Return explicit alignment errors instead of silently shifting paragraphs.
  - Later, a separate repair step may handle malformed candidate structure.

- `src/translation/panel_prompts.py`
  - Judge, paragraph synthesis, consistency audit, and repair prompts.
  - Keep panel prompt evolution separate from the stable single-mode prompts.

- `src/translation/panel_blinding.py`
  - Produce deterministic private option mappings per
    `(run_id, language, paragraph_id, judge_id)`.
  - Randomize option order independently for each judge.
  - Restore option IDs to internal candidate IDs only after parsing.

- `src/translation/panel_aggregation.py`
  - Pure deterministic functions for validation, pairwise victories, Borda
    points, min-max score normalization, critical-error rules, disagreement
    metrics, and synthesis-option selection.
  - Must not call an LLM.

- `src/translation/panel_reporting.py`
  - Persist raw requests/responses, private mappings, aggregates, selected
    options, audits, repairs, and an inspectable Markdown summary.

## Configuration

Add and validate these settings in `TranslationWorkflowConfig`:

```text
EVALUATION_MODE=single|panel|adaptive_panel
PANEL_JUDGES=openai:...,anthropic:...,gemini:...
PANEL_MAX_PARALLEL_JUDGES=3
PANEL_SOURCE_CONTEXT_WINDOW=1
PANEL_TARGET_HISTORY_WINDOW=1
PANEL_PAIRWISE_WEIGHT=0.45
PANEL_RANKING_WEIGHT=0.35
PANEL_SCORE_WEIGHT=0.20
PANEL_CRITICAL_ERROR_CONFIRMATIONS=2
PANEL_SYNTHESIS_TOP_COUNT=2
PANEL_INCLUDE_DIVERSITY_OPTION=1
PANEL_RANDOM_SEED=
```

Initial validation should reject:

- unknown evaluation modes
- fewer than two successful candidate translations in panel mode
- fewer than three configured panel judges
- duplicate judge model families in the initial fixed panel
- weights that do not sum to `1.0`
- unsupported provider/model combinations

Keep `adaptive_panel` accepted only when its thresholds and tie-breaker behavior
are implemented. Until then, fail clearly rather than treating it as `panel`.

## Provider Layer First

The repository currently reserves Anthropic and Gemini candidate names but
raises `NotImplementedError` for those providers. A genuine multi-family panel
therefore depends on implementing a shared provider layer before panel judging.

Use one request shape:

```python
ModelRequest(
    provider=...,
    model=...,
    temperature=...,
    prompt=...,
    image_data_url=...,
    response_format="json",
    cache_namespace=...,
    label=...,
)
```

The provider layer should return text plus request metadata. Provider SDK
details must not leak into panel aggregation or Burr actions.

Refactor existing OpenAI candidate, critic, and final calls onto this layer in a
behavior-preserving change before adding panel mode. This is the main
duplication-prevention step.

## State Contracts

Use stable, JSON-serializable state values. Suggested panel keys:

```text
aligned_candidates[language][paragraph_id]
panel_evaluation_blocks[language][paragraph_id][judge_id]
panel_judge_results[language][paragraph_id][judge_id]
panel_aggregates[language][paragraph_id]
panel_selected_options[language][paragraph_id]
final_paragraphs[language][paragraph_id]
book_audits[language]
repair_results[language][paragraph_id]
final_translations[language]
```

Keep private candidate-option mappings in versioned artifacts, not in prompts
or public reports. They may be present in Burr state if needed for recovery,
but must remain separate from judge-visible blocks.

## Paragraph Alignment Gate

Before any judge call:

1. Split the source with the existing `split_source_paragraphs`.
2. Split every successful candidate using the same paragraph rule.
3. Assign stable paragraph IDs such as `p0001`.
4. Verify every candidate has exactly one aligned option per source paragraph.
5. Record associated spread pages and image references when available.

For the first version, fail panel evaluation for a language when alignment is
invalid and preserve the candidates/artifacts for diagnosis. Do not use fuzzy
alignment in the first implementation; silent paragraph drift would corrupt
every later score.

## Judge Execution

For each language and paragraph:

1. Build a canonical unblinded evaluation block.
2. Blind and independently shuffle candidates for each judge.
3. Run configured judges concurrently, bounded by
   `PANEL_MAX_PARALLEL_JUDGES`.
4. Parse and validate the structured result.
5. Retry once with a schema-correction prompt when validation fails.
6. Store failed judge results explicitly; do not invent neutral scores.

Each judge must receive only:

- the shared rubric
- glossary guidance
- configured nearby source context
- blinded current-paragraph options
- associated image only for configured multimodal judges

Judge identity and candidate identity must not appear in another judge's
request.

The fixed-panel pipeline freezes all judgments before synthesis, so a previous
final target paragraph does not yet exist during judging. Keep previous-final
context in sequential synthesis only. Supplying it to judges would require a
different alternating `judge paragraph -> synthesize paragraph` workflow and
would make later judgments dependent on earlier synthesis choices.

## Deterministic Aggregation

Implement and unit test aggregation independently from prompts and providers:

1. Validate rankings contain every eligible option exactly once.
2. Derive pairwise results from comparisons, rankings, and criterion scores.
3. Compute Borda points.
4. Min-max normalize each judge's scores per criterion.
5. Apply configurable signal weights.
6. Apply confirmed critical-error penalties or vetoes.
7. Compute disagreement metrics.
8. Select top two options plus one meaningfully distinct option when available.
9. Group remarks into consensus, required fixes, critical minority findings,
   stylistic suggestions, recommended phrases, and unresolved disagreements.

The aggregate artifact must show every intermediate value used to reach the
result. An LLM may summarize grouped remarks, but may not alter rankings,
penalties, or selected options.

## Sequential Synthesis And Audit

Generate final paragraphs in source order. Each call receives the frozen
aggregate, selected options, glossary, nearby source context, and configured
previous-final history. Store each completed paragraph immediately so a failed
run can resume without regenerating earlier work.

After assembly:

1. Run one whole-book consistency audit that returns paragraph-scoped findings.
2. Validate audit paragraph IDs and repair instructions.
3. Repair only flagged paragraphs, with neighboring final paragraphs as
   context.
4. Reassemble and write `final_translations`.

The initial audit should not rewrite the whole book. A second final-quality
audit and iterative repair loop can be added after the first version is
measured.

## Artifact Layout

Extend each language run directory without changing current candidate/final
locations:

```text
translation/{language_code}/{run_id}/
  candidates/
  panel/
    alignment.json
    mappings/
    requests/
    raw_judgments/
    aggregates/
    selected_options.json
    audit.json
    repairs.json
    report.md
  final-text.txt
```

Write artifacts after each panel stage. Avoid embedding image data URLs in
artifacts; store image paths or hashes instead.

## Burr Integration

Build the action list and transitions according to `evaluation_mode` while
keeping one shared entrypoint:

```python
if runtime.evaluation_mode == "single":
    actions = single_actions
    transitions = single_transitions
    terminal_action = "generate_final_text"
elif runtime.evaluation_mode == "panel":
    actions = panel_actions
    transitions = panel_transitions
    terminal_action = "finalize_translation"
```

Return the terminal action with the application/configuration or provide a
small `run_application` helper so `src/translation/main.py` does not hard-code
`halt_after=["generate_final_text"]`.

Do not force panel state into the existing `critic_reviews`,
`critic_summaries`, or `critic_winners` shapes. Reuse `final_translations` and
artifact persistence boundaries, not incompatible critic concepts.

## Testing Strategy

Add a real unit-test suite before model-backed integration tests.

### Pure unit tests

- paragraph splitting and exact alignment success/failure
- deterministic blinding with independent judge permutations
- option-ID restoration
- judge schema validation and malformed-result rejection
- min-max normalization, including equal-score behavior
- Borda and pairwise aggregation
- critical-error confirmation and veto behavior
- disagreement metrics and diversity selection
- config parsing and validation

### Workflow tests with fake providers

- `single` mode still follows the current four actions and output shape
- `panel` mode follows the new actions and produces `final_translations`
- judge failure does not erase successful judgments
- synthesis resumes from persisted final paragraphs
- audit repairs only flagged paragraph IDs
- text and multimodal candidate generation both feed the same panel alignment
  contract

### Small live evaluation

Use one target language, three candidates, three judges, and 3-5 selected
paragraphs. Review artifacts manually before running a whole book. Compare
panel results to the current single critic and record human preferences before
calibrating weights or adding `adaptive_panel`.

## Delivery Phases

### Phase 0: Characterization

- Add tests that lock down current `single` behavior, prompt inputs, state keys,
  artifact paths, and final output.
- Add config validation without changing defaults.

### Phase 1: Shared infrastructure

- Extract the provider request/recovery/cache layer.
- Add typed panel models and strict structured-response validation.
- Refactor current OpenAI calls through the provider layer and verify no
  behavior change.

### Phase 2: Fixed panel evaluation

- Add alignment, blinding, judge execution, deterministic aggregation, and
  detailed artifacts.
- Initially stop after aggregation for manual inspection.

### Phase 3: Panel synthesis

- Add sequential paragraph synthesis and final assembly.
- Reuse current final translation persistence.

### Phase 4: Audit and repair

- Add whole-book consistency audit and targeted repairs.
- Extend the panel report with audit decisions and before/after paragraph text.

### Phase 5: Calibration and adaptive panel

- Compare against human-reviewed examples.
- Tune weights, veto thresholds, and disagreement thresholds.
- Add specialist/tie-breaker calls only after fixed-panel behavior is trusted.

## First-Version Boundaries

To keep the addition controlled, the first release should not include:

- fuzzy or LLM-driven paragraph realignment
- connected two-paragraph judging units
- dynamic specialist selection
- multiple audit/repair loops
- judge reliability learning
- automatic weight calibration

These remain compatible extensions once the fixed panel produces reliable,
inspectable artifacts.

## Main Risks

- **Candidate paragraph drift:** mitigate with a strict alignment gate.
- **False multi-model diversity:** require distinct implemented provider
  families.
- **Invalid structured output:** validate, retry once, and record failures.
- **Cost explosion:** cache every request and begin with a paragraph subset.
- **Workflow file growth:** keep actions thin and move domain logic to panel
  modules.
- **Recovery gaps:** persist after each stage and each synthesized paragraph.
- **Single-mode regression:** retain the current path and lock it down with
  characterization tests.

## Recommended First Pull Request

The safest first pull request should contain only:

1. `EVALUATION_MODE` configuration and validation with default `single`.
2. Characterization tests for the current single workflow.
3. A shared provider abstraction supporting OpenAI, Anthropic, and Gemini with
   cache/retry behavior.
4. Refactoring of existing calls onto that abstraction with unchanged outputs.
5. Empty panel module boundaries and typed contracts, but no panel graph yet.

This creates the reusable foundation and proves the existing workflow remains
unchanged before the larger panel behavior lands.
