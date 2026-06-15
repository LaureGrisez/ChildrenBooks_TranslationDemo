# LLM Block Input/Output Examples

This document explains what each orange, LLM-based workflow block actually
receives and returns.

The examples use the French-to-English Barbapapa run from June 12, 2026 where
possible:

- Source: `l_arbre_de_barbapapa_INT.repaired.txt`
- Candidates: `translation/en/20260612_110912/candidates/`
- Critic and summary: exact cached model responses
- Final synthesis: exact generated translation

`REAL` means the excerpt comes from that run. `ILLUSTRATIVE` means it follows
the implemented prompt and output schema, but no corresponding panel artifact
has been generated yet.

## The Main Distinction

The **Text only** and **Multimodal** single-critic workflows currently contain
three evaluation/finalization model calls:

1. **Single critic**: judges the candidate translations and returns a detailed
   structured review.
2. **Critic summary**: asks an LLM to condense that review into editorial
   instructions.
3. **Final synthesis**: writes the final translation using the source,
   candidates, and condensed instructions.

Your intuition is reasonable: the critic already returns `revision_instructions`
and a `concise_summary`, so the separate **Critic summary** call is not strictly
necessary. It is currently an additional LLM condensation step, not
deterministic processing.

The **Panel judging** workflow does not use this critic-summary call. Its
processing between judging and synthesis is deterministic aggregation.

## Shared French-to-English Example

The following small excerpt is reused below:

```text
French source:
C’est trop difficile… Ils n’y arriveront pas !
Mais pourquoi construire un pont ? Avez-vous oublié
que les Barbapapas peuvent changer de forme ?

Candidate gpt4o:
"It's too difficult... They won't make it!
But why build a bridge? Have you forgotten
that the Barbapapas can change shape?"

Candidate gpt5_5:
It’s too hard… They’ll never manage it!

But why build a bridge? Have you forgotten
that the Barbapapas can change shape?
```

The important difference is that `gpt4o` incorrectly places the narrator's
text inside quotation marks, while `gpt5_5` splits one source paragraph into
two target paragraphs.

---

## Mode: Text Only

### Generate Text Candidates

**Purpose:** Produce several complete alternative translations with different
translation stances.

This block may contain both LLM calls and a non-LLM machine translation call.
For example, `gpt4o` and `gpt5_5` are LLM-based, while the configured Google
translation candidate is an external translation service.

**Representative input to the `gpt5_5` candidate**

```text
Translate a Barbapapa children's book from French into English.

Audience and style:
- Children aged 5-8
- Warm, simple, playful
- Easy to read aloud
- Preserve paragraph breaks and quoted speech

Translator stance:
- Slightly more playful and lively while preserving every scene.

Required English names include:
Claudine -> Cindy
François -> Frank
Barbidou -> Barbazoo

Source:
C’est trop difficile… Ils n’y arriveront pas !
Mais pourquoi construire un pont ? Avez-vous oublié
que les Barbapapas peuvent changer de forme ?
```

**Output - REAL**

```text
It’s too hard… They’ll never manage it!

But why build a bridge? Have you forgotten
that the Barbapapas can change shape?
```

Each candidate call returns a complete translated book as plain text. Workflow
code then records metadata such as candidate name, provider, model, status,
latency, and error.

### Single Critic

**Purpose:** Judge all completed candidate books in one model call.

This is the actual **judging** stage in Text only mode.

**Representative input**

```text
Role: senior editor for translated picture books

Evaluate:
- faithfulness
- child-friendly phrasing
- read-aloud rhythm
- character names
- cultural fit
- quotes, imagery, and paragraph breaks

French source:
C’est trop difficile… Ils n’y arriveront pas !
Mais pourquoi construire un pont ? Avez-vous oublié
que les Barbapapas peuvent changer de forme ?

Candidate gpt4o:
"It's too difficult... They won't make it!
But why build a bridge? Have you forgotten
that the Barbapapas can change shape?"

Candidate gpt5_5:
It’s too hard… They’ll never manage it!

But why build a bridge? Have you forgotten
that the Barbapapas can change shape?

Return JSON containing an overall winner, ranking, paragraph analysis,
candidate assessments, revision instructions, and concise summary.
```

**Output excerpt - REAL**

```json
{
  "overall_winner": "gpt5_5",
  "ranking": ["gpt5_5", "gpt4o", "google_translation"],
  "paragraph_analysis": [
    {
      "paragraph_number": 3,
      "best_candidate": "gpt5_5",
      "notes": "gpt5_5 clearly wins because it does not put narration into dialogue. gpt4o incorrectly encloses the whole passage in quotation marks. gpt5_5 does split the source paragraph into two, which should be corrected."
    }
  ],
  "revision_instructions": [
    "Use gpt5_5 as the base translation, but restore the French paragraph breaks.",
    "Do not put narration inside quotation marks."
  ],
  "concise_summary": "gpt5_5 is the strongest base: it is natural, warm, and child-friendly. Revise mainly for paragraph fidelity and a few small wording refinements."
}
```

### Critic Summary

**Purpose:** Turn the detailed critic JSON into shorter editorial guidance for
the final translator.

This is another LLM call. It does not judge candidates again and it does not
translate. It mostly reformats and condenses the previous critic output.

**Input**

```text
Summarize this translation critique for the final English translator.
Focus on decisions, tradeoffs, and concrete revision instructions.

Critique:
{
  "overall_winner": "gpt5_5",
  "ranking": ["gpt5_5", "gpt4o", "google_translation"],
  "paragraph_analysis": [...],
  "candidate_assessment": [...],
  "revision_instructions": [...],
  "concise_summary": "..."
}
```

**Output excerpt - REAL**

```text
Use gpt5_5 as the base translation. It is the best overall: faithful,
warm, natural for a read-aloud picture book, and consistently child-friendly.
It also avoids gpt4o’s major error of putting narration inside quotation marks.

Concrete revision instructions:
1. Preserve the French paragraph structure.
2. Do not put narration inside quotation marks.
3. Keep all required English names consistent.
4. Change "carry it away" to "carry it" and replace "plant things".
```

**Architectural note:** This output could instead be produced by deterministic
code selecting fields such as `revision_instructions`, `decision_reasoning`,
and `concise_summary` from the critic JSON. Removing this extra LLM call would
make the workflow closer to **judging -> synthesis**.

### Final Synthesis

**Purpose:** Write one final coherent English book, using the candidates as
draft material and the critic summary as editorial instructions.

This is the actual **synthesis** stage in Text only mode.

**Representative input**

```text
Create the final English translation.

French source:
C’est trop difficile… Ils n’y arriveront pas !
Mais pourquoi construire un pont ? Avez-vous oublié
que les Barbapapas peuvent changer de forme ?

Candidate gpt4o:
"It's too difficult... They won't make it!
But why build a bridge? Have you forgotten
that the Barbapapas can change shape?"

Candidate gpt5_5:
It’s too hard… They’ll never manage it!

But why build a bridge? Have you forgotten
that the Barbapapas can change shape?

Critic summary:
- Use gpt5_5 as the base.
- Preserve the French paragraph structure.
- Do not put narration inside quotation marks.
```

**Output excerpt - REAL**

```text
It’s too hard… They’ll never manage it! But why build a bridge? Have you
forgotten that the Barbapapas can change shape?
```

The synthesis keeps `gpt5_5`'s natural wording, removes the incorrect quotation
marks, and recombines the text into one paragraph to match the source.

---

## Mode: Multimodal

Multimodal mode uses the same single-critic architecture, but candidate
generation sees spread images and final synthesis is performed one source
paragraph at a time.

### Generate Multimodal Candidates

**Purpose:** Translate each double-page spread while using both its French text
and its textless illustration.

**Representative input**

```text
Model inputs:
- Textless image of the current double-page spread
- French text for the spread
- Previous French source context
- Recently translated English spread context
- Character-name glossary
- Candidate stance

Current French source:
Les voilà bientôt dans l’île avec leur panier.
Sur cette île, il y a un très gros arbre qui sert de maison à une chouette.

Instruction:
Return only the English translation of the current source pages.
```

**Output - ILLUSTRATIVE, based on the real candidate**

```text
Before long, there they are on the island with their basket.
On the island, there is a very big tree that is home to an owl.
```

The image is sent directly to the multimodal model alongside the prompt. The
spread outputs are joined to form each complete candidate book.

### Single Critic

**Purpose:** Judge the completed multimodal candidate books.

**Input and output shape:** The same as **Text only -> Single Critic**. The
critic receives the French source and completed candidate texts. It does
**not** receive the spread images in the current implementation.

**Output example - REAL single-critic excerpt**

```json
{
  "overall_winner": "gpt5_5",
  "revision_instructions": [
    "Use gpt5_5 as the base translation.",
    "Restore the French paragraph breaks.",
    "Do not put narration inside quotation marks."
  ]
}
```

### Critic Summary

**Purpose:** Condense the detailed critic review.

**Input and output shape:** Exactly the same as
**Text only -> Critic Summary**. This is also a separate LLM call and could be
replaced by deterministic extraction from the critic JSON.

### Final Synthesis

**Purpose:** Create the final translation one source paragraph/PDF text page at
a time while preserving page alignment.

Unlike multimodal candidate generation, final synthesis does **not** directly
receive the illustration in the current implementation. It uses the
image-informed candidate books as editorial references.

**Representative input for one paragraph**

```text
Create only paragraph 2 of 22.
Return exactly one paragraph.

Previous French source:
Les plus belles mûres sont dans l’île...

Current French source:
C’est trop difficile… Ils n’y arriveront pas !
Mais pourquoi construire un pont ? Avez-vous oublié
que les Barbapapas peuvent changer de forme ?

Next French source:
Les voilà bientôt dans l’île avec leur panier...

Previous final English paragraph:
It’s the end of summer...

Candidate translation books:
- gpt4o complete translation
- gpt5_5 complete translation

Critic summary:
- Use gpt5_5 as the base.
- Preserve paragraph alignment.
- Do not put narration inside quotation marks.
```

**Output - REAL final-text excerpt**

```text
It’s too hard… They’ll never manage it! But why build a bridge? Have you
forgotten that the Barbapapas can change shape?
```

---

## Mode: Panel Judging

Panel mode replaces the single whole-book critic with multiple independent,
blinded paragraph judges. It then combines their judgments using deterministic
Python code before synthesis.

The examples below are `ILLUSTRATIVE` because there are currently no persisted
panel-run artifacts in `translation/`.

### Generate Candidates

**Purpose:** Produce alternative complete translations.

**Input and output shape:** The same as **Text only -> Generate Text
Candidates**. Candidate generation is not paragraph-by-paragraph in text panel
mode.

### Independent Judge Panel

**Purpose:** Have several independent models judge each paragraph without
seeing candidate identities or providers.

Before this LLM block, deterministic code aligns candidate paragraphs and
renames them independently for each judge, for example `option_a` and
`option_b`.

**Representative input to one judge**

```text
Previous French paragraph:
Les plus belles mûres sont dans l’île...

Current French paragraph:
C’est trop difficile… Ils n’y arriveront pas !
Mais pourquoi construire un pont ? Avez-vous oublié
que les Barbapapas peuvent changer de forme ?

Next French paragraph:
Les voilà bientôt dans l’île avec leur panier...

Blinded options:
{
  "option_a": "\"It's too difficult... They won't make it! But why build a bridge?...\"",
  "option_b": "It’s too hard… They’ll never manage it! But why build a bridge?..."
}

Score every option from 0 to 10 for faithfulness, naturalness,
child-friendliness, read-aloud quality, continuity, glossary compliance,
and structure. Return JSON only.
```

**Output - ILLUSTRATIVE**

```json
{
  "comparisons": [
    {
      "criterion": "faithfulness",
      "preferred_options": ["option_b"],
      "evidence": "option_a incorrectly presents narrator text as dialogue."
    }
  ],
  "option_scores": {
    "option_a": {
      "faithfulness": 6,
      "naturalness": 7,
      "child_friendliness": 7,
      "read_aloud": 6,
      "continuity": 6,
      "glossary_compliance": 10,
      "structure": 8,
      "critical_errors": ["Narration incorrectly placed inside quotation marks"],
      "remarks": ["Faithful wording, but the speaker attribution is wrong"]
    },
    "option_b": {
      "faithfulness": 9,
      "naturalness": 9,
      "child_friendliness": 9,
      "read_aloud": 9,
      "continuity": 8,
      "glossary_compliance": 10,
      "structure": 6,
      "critical_errors": [],
      "remarks": ["Natural wording, but split into two paragraphs"]
    }
  },
  "overall_ranking": ["option_b", "option_a"],
  "recommended_phrases": [
    {
      "option": "option_b",
      "phrase": "It’s too hard… They’ll never manage it!",
      "reason": "Natural and child-friendly"
    }
  ],
  "confidence": 0.92
}
```

After every judge responds, deterministic code restores candidate identities
and aggregates pairwise preference, ranking, normalized criterion scores,
confirmed critical errors, repeated remarks, and disagreement. This
**Deterministic aggregation** block is not LLM-based.

### Sequential Paragraph Synthesis

**Purpose:** Generate the final translation one paragraph at a time using
deterministically aggregated panel guidance.

**Representative input**

```text
Previous final English paragraph:
The best blackberries are on the island, but how can they get there?
“Let’s build a bridge,” says Cindy.

Current French paragraph:
C’est trop difficile… Ils n’y arriveront pas !
Mais pourquoi construire un pont ? Avez-vous oublié
que les Barbapapas peuvent changer de forme ?

Selected candidate options:
{
  "gpt5_5": "It’s too hard… They’ll never manage it! But why build a bridge?...",
  "gpt4o": "\"It's too difficult... They won't make it! But why build a bridge?...\""
}

Frozen panel guidance:
{
  "ranking": ["gpt5_5", "gpt4o"],
  "confirmed_critical_errors": {
    "gpt4o": ["narration incorrectly placed inside quotation marks"],
    "gpt5_5": []
  },
  "consensus_remarks": {
    "gpt5_5": ["natural wording, but preserve source paragraph structure"]
  },
  "recommended_phrases": [...]
}
```

**Output - ILLUSTRATIVE, matching the real final**

```text
It’s too hard… They’ll never manage it! But why build a bridge? Have you
forgotten that the Barbapapas can change shape?
```

### Whole-Book Audit

**Purpose:** Check the assembled final book for cross-paragraph consistency
problems. The model must identify repairs, not rewrite the book.

**Representative input**

```text
Audit the complete English children's-book translation.

Check:
- character names and terminology
- narrative voice and tense
- repeated phrases and transitions
- read-aloud rhythm
- quotes and paragraph structure
- source meaning

French source:
[complete French book]

Final English translation:
[all synthesized English paragraphs]

Return JSON findings with paragraph IDs and repair instructions.
```

**Output - ILLUSTRATIVE**

```json
{
  "findings": [
    {
      "paragraph_id": "p0008",
      "severity": "minor",
      "instruction": "Remove the added directional meaning in 'carry it away'; use 'carry it'."
    },
    {
      "paragraph_id": "p0014",
      "severity": "minor",
      "instruction": "Replace 'plant things' with a natural child-friendly phrase such as 'plant some greenery'."
    }
  ]
}
```

An empty `findings` list means no repair calls are needed.

### Targeted Repairs

**Purpose:** Rewrite only paragraphs explicitly flagged by the whole-book
audit.

**Representative input**

```text
Source paragraph:
Il ne reste plus qu’à planter pour retenir la terre et faire joli.

Previous final paragraph:
There we go! The otters can move into their new burrow.

Current final paragraph:
Now they just need to plant things to hold the soil in place and make it look pretty.

Next final paragraph:
Everything is back to normal now.

Audit finding:
Replace "plant things" with a natural child-friendly phrase such as
"plant some greenery".

Return only the repaired paragraph.
```

**Output - ILLUSTRATIVE, matching the real final wording**

```text
Now they just need to plant some greenery to hold the soil in place and make
it look pretty.
```

Paragraphs without audit findings pass through unchanged. Deterministic code
then joins all repaired and unchanged paragraphs into the final book.

---

## Summary By Mode

| Mode | LLM-based blocks | Non-LLM processing between them |
| --- | --- | --- |
| Text only | Candidate generation, single critic, critic summary, final synthesis | JSON parsing, reference normalization, persistence |
| Multimodal | Multimodal candidate generation, single critic, critic summary, paragraph-aligned final synthesis | Text/spread alignment, image rendering, paragraph normalization |
| Panel judging | Candidate generation, independent judges, paragraph synthesis, whole-book audit, targeted repairs | Paragraph alignment, blinding, identity restoration, judgment aggregation, final assembly |

The main simplification opportunity is in the single-critic modes:

```text
Current:
judging LLM -> summary LLM -> synthesis LLM

Possible:
judging LLM -> deterministic extraction of guidance -> synthesis LLM
```
