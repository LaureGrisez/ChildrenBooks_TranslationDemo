# Panel Evaluation Mode Direction

## Purpose

The current workflow generates multiple translation candidates, but their
diversity eventually passes through one critic, one critic-summary step, and one
final synthesizer. This creates a judgment bottleneck and can introduce
self-preference when a model judges a translation produced by the same model
family.

The proposed `panel` evaluation mode should make evaluation more robust by:

- using independent judges from different model families
- hiding candidate model and provider identities
- comparing all candidate versions of the same paragraph together
- aggregating rankings, scores, evidence, and critical errors
- generating the final translation paragraph by paragraph with continuity
  context
- auditing the completed translation at book level

Candidate-generation mode and evaluation mode should remain independent:

```text
WORKFLOW_MODE=text | multimodal
EVALUATION_MODE=single | panel | adaptive_panel
```

`WORKFLOW_MODE` controls how candidates are generated. `EVALUATION_MODE`
controls how candidates are judged and synthesized.

## Recommended Direction

Use paragraph-level comparative judging as the primary evaluation method, then
use whole-book passes for consistency and final-quality auditing.

```text
Generate candidate translations
        |
Split and align candidates by paragraph
        |
Run independent multi-model judge panel per paragraph
        |
Normalize and aggregate judgments deterministically
        |
Select top candidates and aggregate remarks
        |
Generate one final paragraph using previous-final context
        |
Assemble final book
        |
Run whole-book consistency and quality audits
        |
Repair only flagged paragraphs
```

Paragraph evaluation gives precise and explainable decisions. Whole-book
evaluation catches problems that cannot be judged locally, such as inconsistent
voice, terminology, repetition, and narrative flow.

## Why Compare Candidates Paragraph By Paragraph

Each judge should see all candidate versions of the current paragraph at once.
Models are generally more reliable at identifying relative differences than at
assigning meaningful absolute scores to isolated translations.

Comparative evaluation allows a judge to make claims such as:

- Option B preserves an action that Option A omits.
- Option A is more natural than Option C.
- Option C has the strongest read-aloud rhythm.
- Option B is most faithful but needs wording borrowed from Option A.

The judge should rank and score only the current paragraph candidates. Limited
surrounding context should be supplied to support continuity, but it should not
expand the object being ranked.

## Paragraph Evaluation Block

For each source paragraph, construct one evaluation block containing:

1. Evaluation rubric and output schema
2. Target-language glossary and established terminology
3. Previous source paragraph
4. Current source paragraph
5. Next source paragraph
6. Previous approved or generated target paragraph
7. Blinded candidate versions of the current paragraph
8. Associated spread image when using multimodal judging

The exact context window should remain configurable and be validated
experimentally. A strong initial default is one previous and one next source
paragraph, plus the previous final target paragraph.

For two-page spread segments, the evaluation unit may contain two paragraph
blocks when the meaning or visible action strongly connects them. The returned
judgment must still identify findings and rankings per paragraph.

## Independent Multi-Model Panel

The panel should contain judges from genuinely different model families, for
example:

- one OpenAI judge
- one Anthropic judge
- one Gemini judge
- an optional target-language specialist
- an optional multimodal visual-grounding judge

Every judge evaluates independently and must not see other judges' outputs.
Using multiple prompts with the same model can add perspective, but it should
not be treated as equivalent to model-family diversity.

### Candidate Blinding

Before each judge call:

- remove candidate names, providers, models, temperatures, and stances
- assign neutral identifiers such as `option_k`, `option_m`, and `option_r`
- randomize option order independently for every judge and paragraph
- retain a private mapping so results can be restored to candidate identities

Independent randomization reduces model-name bias and positional bias.

## Judge Responsibilities

Every core judge should evaluate the same shared criteria:

- faithfulness to the source
- natural target-language phrasing
- child-friendly language
- read-aloud quality
- continuity with nearby paragraphs
- glossary and character-name compliance
- preservation of quotes, paragraph structure, and meaning

Specialist judges may emphasize one area:

- **Faithfulness judge:** omissions, additions, mistranslations, and narrative
  accuracy
- **Native-language editor:** fluency, grammar, age suitability, and cultural
  fit
- **Read-aloud judge:** rhythm, repetition, sentence length, and oral clarity
- **Visual-grounding judge:** agreement with visible characters, actions, and
  scene details
- **Structure judge:** paragraph boundaries, quotes, page order, and glossary
  consistency

The shared rubric makes outputs aggregatable. Specialist roles create useful
perspective diversity.

## Judge Output Contract

Judges should identify evidence before assigning scores. Scores unsupported by
comparative evidence should receive less weight or be rejected.

```json
{
  "comparisons": [
    {
      "criterion": "faithfulness",
      "preferred_options": ["option_m"],
      "evidence": "Option K omits the character's transformation."
    }
  ],
  "option_scores": {
    "option_k": {
      "faithfulness": 7,
      "naturalness": 9,
      "read_aloud": 9,
      "continuity": 8,
      "glossary_compliance": 10,
      "critical_errors": [],
      "remarks": ["Warm and rhythmic, but missing one visible action."]
    }
  },
  "overall_ranking": ["option_m", "option_k", "option_r"],
  "recommended_phrases": [
    {
      "option": "option_k",
      "phrase": "...",
      "reason": "Best read-aloud rhythm."
    }
  ],
  "confidence": 0.85
}
```

The contract separates:

- comparative evidence
- criterion scores
- overall ranking
- critical errors
- actionable remarks
- useful phrases
- judge confidence

## Aggregation Strategy

Aggregation should be primarily deterministic. An LLM may summarize aggregated
evidence afterward, but it should not silently replace the panel's results with
its own preference.

### Primary Signals

Use three complementary signals:

1. **Pairwise comparison victories**

   Derive candidate-versus-candidate results for each criterion from each
   judge's comparisons and scores. Aggregate these across judges using majority
   voting or a Condorcet-style method.

2. **Ranking position**

   Convert each judge's ranking into Borda-count points. Rankings are useful
   because they are less sensitive to differences in score scale.

3. **Normalized criterion scores**

   Normalize each judge's scores within the current paragraph and criterion.
   This prevents a generous judge's `9` and a strict judge's `7` from being
   interpreted as directly comparable absolute measurements.

An initial experimental weighting could be:

```text
45% pairwise comparison results
35% ranking position
20% normalized criterion scores
```

These weights should be calibrated using human-reviewed examples rather than
treated as permanent.

### Score Normalization

A simple initial approach is min-max normalization within each judge,
paragraph, and criterion:

```text
normalized_score =
    (score - lowest_option_score)
    / (highest_option_score - lowest_option_score)
```

When all options receive the same score, that criterion contributes no
preference for that judge.

Z-score normalization may become useful with more candidates and calibration
data, but min-max normalization is easier to inspect during the first
implementation.

### Critical Errors And Vetoes

Critical errors must not disappear inside an average score. Examples include:

- omitted or invented action
- incorrect character or speaker
- contradiction with the source or image
- incorrect required character name
- missing paragraph or broken page ordering

A confirmed critical error should apply a substantial penalty or make an option
ineligible to win. Confirmation may require agreement from two judges, or one
high-confidence specialist judge with concrete evidence.

### Judge Confidence And Disagreement

Confidence should influence how strongly a judgment contributes, but it should
not overpower concrete evidence.

Store disagreement metrics such as:

- variance of normalized scores
- number of different first-place choices
- pairwise cycles
- conflicting interpretations of the source

High disagreement should trigger an additional judge, a specialist review, or a
deeper evaluation block. It should not be hidden behind one aggregate number.

## Aggregating Remarks

Do not concatenate every judge remark directly into the synthesis prompt.
Instead, group findings into:

- **Consensus findings:** independently identified by multiple judges
- **Required fixes:** factual, structural, or glossary problems
- **Critical minority findings:** serious evidence-backed issues raised by one
  judge
- **Stylistic suggestions:** optional improvements
- **Recommended phrases:** particularly successful wording
- **Unresolved disagreements:** conflicting interpretations that need caution

An LLM may summarize these already-grouped findings into concise synthesis
guidance. Candidate and judge model identities should remain hidden.

## Selecting Candidates For Final Synthesis

The final paragraph generator does not need every candidate. Supply:

```text
Top 2 options by aggregate result
+
1 useful diversity option
```

The diversity option should contribute something distinct, such as:

- the best read-aloud phrasing
- the strongest visual grounding
- a useful phrase recommended by multiple judges
- a valid interpretation not represented by the top two

This is preferable to selecting the top three scores when those options are
nearly identical.

## Paragraph Synthesis Block

Generate the final translation sequentially, one paragraph at a time. For each
paragraph, provide:

1. Current source paragraph
2. Previous and next source context
3. Previous final target paragraph, optionally the previous two
4. Glossary and established style decisions
5. Selected candidate paragraph options
6. Aggregate rankings and criterion results
7. Consensus strengths
8. Required fixes and critical errors to avoid
9. Recommended phrases
10. Unresolved disagreements
11. Associated spread image when useful and supported

The generator returns only the current final paragraph. Sequential generation
allows earlier approved paragraphs to become continuity context for later
paragraphs.

## Whole-Book Passes

Paragraph-level decisions alone cannot guarantee a coherent book. After
assembling all final paragraphs, run whole-book checks.

### Consistency Audit

Check:

- character names and terminology
- narrative voice and tense
- repeated phrases and callbacks
- transitions between paragraphs
- read-aloud rhythm across the book
- quote and paragraph structure

### Final Quality Audit

Compare the assembled final translation against the complete source and, in
multimodal mode, relevant images. The audit should flag specific paragraphs and
provide repair instructions rather than rewrite the entire book.

Only flagged paragraphs should be regenerated. Their neighboring final
paragraphs should be supplied as context so repairs do not damage continuity.

## Adaptive Panel Mode

A full panel for every paragraph may be expensive. `adaptive_panel` can reduce
cost while retaining stronger review where it matters:

1. Run three core judges.
2. Accept paragraphs with strong agreement and no critical errors.
3. Call specialist or tie-breaker judges for high-disagreement paragraphs.
4. Apply deeper review to paragraphs flagged by the whole-book audit.

This mode should use explicit thresholds and log why additional judges were or
were not called.

## Suggested Workflow Stages

```text
generate_candidates
align_candidate_paragraphs
build_blinded_evaluation_blocks
run_panel_judges
normalize_and_aggregate_judgments
aggregate_judge_remarks
select_synthesis_options
generate_final_paragraphs
audit_book_consistency
audit_final_quality
repair_flagged_paragraphs
```

Artifacts should preserve:

- blinded judge requests
- private candidate-ID mappings
- raw judge responses
- normalized scores
- pairwise and ranking results
- disagreement metrics
- aggregated remarks
- selected synthesis options
- audit findings and repairs

This makes every final paragraph traceable and allows later evaluation of judge
reliability.

## Initial Implementation Scope

A practical first panel version should include:

- three judges from different model families
- paragraph-level blinded comparative evaluation
- independently randomized candidate order
- shared structured JSON output
- pairwise, Borda-ranking, and normalized-score aggregation
- explicit critical-error handling
- top two plus one diversity option for synthesis
- sequential paragraph generation using previous-final context
- one whole-book consistency audit
- detailed evaluation artifacts and reports

The first version should keep aggregation formulas configurable and visible.
Human review of a small set of paragraphs should then be used to calibrate
weights, veto thresholds, context windows, and disagreement triggers.

## Open Questions To Validate

- Should the evaluation unit always be one paragraph, or sometimes a connected
  two-paragraph spread?
- How much previous final-target context improves continuity without anchoring
  the judge too strongly?
- Should all judges use the same rubric weights, or should specialist weights
  differ?
- What evidence threshold confirms a critical error?
- Which aggregation method best matches human editor preferences?
- How should diversity-option selection be measured?
- When does visual judging materially improve results?
- At what disagreement threshold is an additional judge worth the cost?

The central design principle is:

> Judges evaluate independently, all candidate versions of the current
> paragraph are compared together, aggregation remains inspectable, and final
> synthesis begins only after judgments are frozen.
