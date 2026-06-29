# Children Books Translation Demo

This repo contains several workflow experiments for children's-book translation.
The most advanced Burr workflow now translates the repaired French Barbapapa
text and uses the CSV glossary to guide character names across languages.

## Requirements

- Python 3.10 or later
- A local virtual environment at `.venv`
- Dependencies installed in that environment
- A `.env` file containing `OPENAI_API_KEY=your_api_key_here`

Example setup:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -U pip litellm openai pillow langgraph langchain-openai agent-framework-openai burr crewai python-dotenv rich deepl pymupdf
.venv/bin/python -m pip install -r requirements-ui.txt
```

## Run The Streamlit Test UI

The Streamlit app runs the actual translation workflows and is intended for
testing text-only, multimodal, and panel-judging behavior:

```bash
.venv/bin/streamlit run streamlit_app.py
```

The repository configures Streamlit to use:

```text
http://[::1]:8511
```

This dedicated loopback address avoids VS Code's automatic port-forwarding
helper, which can occupy `127.0.0.1:8501` and leave `localhost:8501` loading
indefinitely. Confirm the app server is healthy with:

```bash
curl 'http://[::1]:8511/_stcore/health'
```

It should return `ok`.

The UI supports:

- independently selecting text-only or multimodal candidate generation
- independently selecting single-judge or panel-judge evaluation
- selecting the target language from the character glossary
- selecting two to five candidate strategies and configuring their model names
- applying one candidate temperature to all selected candidates
- uploading cleaned source text and, for multimodal mode, the source PDF
- viewing the selected workflow as a Mermaid graph, with colored model-call
  blocks and hover descriptions for every stage
- inspecting generated candidates, per-judge panel scores, aggregate rankings,
  audits, and targeted repairs
- comparing candidates, pre-audit panel synthesis, and final translations using
  the existing comparison-report code
- downloading the final translated text

Panel mode requires credentials only for the providers used by its selected
candidates and judges. Google Translate candidates additionally require the
existing Google Cloud translation credentials.

With **Multimodal + Panel judges**, candidate models receive the rendered PDF
spread images. The panel judges evaluate the resulting aligned candidate text;
they do not currently receive the images.

## Run The Advanced Burr Workflow

From the project root, launch one run with:

```bash
TARGET_LANGUAGES=Finnish BURR_STORAGE_DIR=/private/tmp/.burr .venv/bin/python main.py
```

To run only the first two candidates in the default incremental order
(`google_translation`, then `gpt4o`), use:

```bash
TARGET_LANGUAGES=Finnish MAX_PARALLEL_CANDIDATES=2 BURR_STORAGE_DIR=/private/tmp/.burr .venv/bin/python main.py
```

To choose candidates explicitly, use `CANDIDATE_NAMES`:

```bash
TARGET_LANGUAGES=Finnish CANDIDATE_NAMES=google_translation,gpt4o BURR_STORAGE_DIR=/private/tmp/.burr .venv/bin/python main.py
```

No extra flag is needed for cache-backed recovery. The workflow uses the
response cache automatically by default.

If you want to force a fresh uncached run, use:

```bash
TARGET_LANGUAGES=Finnish BURR_STORAGE_DIR=/private/tmp/.burr TRANSLATION_CACHE=0 .venv/bin/python main.py
```

To run the first multimodal configuration, use:

```bash
WORKFLOW_MODE=multimodal TARGET_LANGUAGES=Finnish BURR_STORAGE_DIR=/private/tmp/.burr .venv/bin/python main.py
```

To evaluate candidates with a blinded multi-model judge panel, use:

```bash
EVALUATION_MODE=panel \
TARGET_LANGUAGES=Finnish \
BURR_STORAGE_DIR=/private/tmp/.burr \
.venv/bin/python main.py
```

By default, every selected LLM candidate model also acts as a judge. Google
Translate remains a candidate only because it cannot perform structured
judging. To override the derived judge panel from the command line, set
`PANEL_JUDGES`, for example:

```bash
PANEL_JUDGES=openai:gpt-4o,anthropic:claude-sonnet-4-6 \
EVALUATION_MODE=panel \
.venv/bin/python main.py
```

Panel mode requires credentials for every selected candidate and judge provider.
All LLM providers are called through LiteLLM; separate Anthropic and Gemini
Python SDKs are not required.

Notes:

- Replace `Finnish` with any language present in `Noms barbapapas - Sheet1.csv`.
- If `TARGET_LANGUAGES` is omitted, the workflow uses all CSV languages except `French`.
- Candidate generation follows this default incremental order: `google_translation`, `gpt4o`, `gpt5_5`, `claude_sonnet_4_6`, `gemini_3`.
- `MAX_PARALLEL_CANDIDATES` limits how many candidates from that order are used.
- `CANDIDATE_NAMES` lets you choose the exact candidate set explicitly.
- `claude_sonnet_4_6` is an Anthropic candidate and requires `ANTHROPIC_API_KEY`. In multimodal mode, its spread request includes the rendered image.
- Up to five unique built-in candidate strategies are supported: Google
  Translate, two OpenAI strategies, Anthropic Sonnet, and Google Gemini.
- The default source text is `l_arbre_de_barbapapa_INT.repaired.txt`.
- Final translations are also exported as plain text under `translation/`, for example `translation/l_arbre_de_barbapapa_INT_fi.txt`.
- Each run also writes versioned artifacts under `translation/{language_code}/{run_id}/`, including `candidates/`, the final text, and a Markdown comparison report.
- LLM and external translation responses are cached under `.translation_cache/`, so rerunning the same job can recover from transient failures without recomputing earlier successful steps.
- LiteLLM provider calls retry transient connection, rate-limit, and server errors automatically. You can tune this with `OPENAI_RETRY_ATTEMPTS` and `OPENAI_RETRY_BASE_DELAY_SECONDS`; the legacy variable names are retained for compatibility.
- `WORKFLOW_MODE=text` is the default. `WORKFLOW_MODE=multimodal` switches candidate generation to spread-aligned translation using the source PDF plus local source-text context.
- `EVALUATION_MODE=single` is the default and preserves the existing critic, critic-summary, and full-book final synthesis stages.
- `EVALUATION_MODE=panel` performs exact paragraph alignment, independently blinded multi-family judging, deterministic aggregation, sequential paragraph synthesis, a whole-book consistency audit, and targeted repairs.
- Panel artifacts are written under `translation/{language_code}/{run_id}/panel/`, including alignment, private mappings, raw judge results, aggregates, audit findings, and repairs.
- When `PANEL_JUDGES` is omitted, panel judges are derived from the selected LLM candidates. The Streamlit panel-judging mode shows this default as editable judge rows, with a judge count plus provider and model fields.
- The Streamlit judge block mirrors candidate configuration: judge count, one shared judge temperature, and one model-selection row per judge. Google Translate is excluded from judge choices.
- `PANEL_JUDGE_TEMPERATURE` controls all panel judgment calls from the command line and defaults to `0.1`.
- A panel must contain at least two distinct judge models. `PANEL_JUDGES` overrides the candidate-derived default and accepts LiteLLM `provider:model` references such as `openai:gpt-4o`, `anthropic:claude-sonnet-4-6`, or `zai:glm-4.5`.
- Gemini candidates and judges use `GEMINI_API_KEY` or `GOOGLE_API_KEY`. The default model is `gemini-2.5-flash`; override it with `GEMINI_MODEL` or the Streamlit model field.
- `DEFAULT_CRITIC_MODEL`, `DEFAULT_AGGREGATION_MODEL`, and `DEFAULT_CRITIC_SUMMARIZER_MODEL` accept `provider:model` values. Bare model names remain compatible and default to OpenAI.
- `DEFAULT_CRITIC_MODEL` controls the single critic and panel whole-book audit. `DEFAULT_AGGREGATION_MODEL` controls final synthesis and targeted repairs; panel score aggregation itself remains deterministic. `DEFAULT_CRITIC_SUMMARIZER_MODEL` controls the single-mode critic-summary call.
- Panel aggregation defaults to 45% pairwise results, 35% ranking position, and 20% normalized criterion scores. Configure these with `PANEL_PAIRWISE_WEIGHT`, `PANEL_RANKING_WEIGHT`, and `PANEL_SCORE_WEIGHT`; they must sum to `1.0`.
- Panel mode uses strict paragraph alignment. It stops with an explicit error when a successful candidate does not preserve the source paragraph count.
- In the current multimodal version, candidate generation uses the textless spread images, while the critic and final synthesizer remain text-only.
- In multimodal mode, each new segment also receives the previously translated target-language segment history. `TARGET_HISTORY_WINDOW=1` is the default.
- Multimodal mode groups consecutive body pages by double-page spread. When both pages in a spread contain text, they are translated together in one multimodal call and returned in page order.
- Multimodal mode assumes the cleaned French text has one paragraph per non-empty body page in the source PDF. For `l_arbre_de_barbapapa`, that mapping is currently valid and currently groups into 16 spread calls.
- `SOURCE_CONTEXT_WINDOW=0` is now the default in multimodal mode, so the current French paragraph stays primary. You can raise it if you want some previous raw French context as well.
- You can tune multimodal context with `TARGET_HISTORY_WINDOW` and `SOURCE_CONTEXT_WINDOW`, and spread rendering size with `MULTIMODAL_IMAGE_DPI`.
- The runnable app entrypoint is `main.py`; the translation code itself lives under `src/translation/`.

## Test Multimodal Image Understanding

To evaluate whether the multimodal models understand the visible action in one
double-page spread with only minimal book context, run:

```bash
.venv/bin/python test/multimodal_action_summary.py \
  --spread-pages 10,11 \
  --models gpt-4o,gpt-5.5 \
  --detail low \
  --dpi 110 \
  --save-input-image
```

This test:

- renders the requested PDF spread as a textless image
- sends a minimal Barbapapa context as the system prompt
- asks each listed model for a short action summary
- writes a Markdown report under `translation/_multimodal_debug/evals/`

The default system prompt lives in:

```text
test/prompts/multimodal_action_summary_system_prompt.txt
```

There is also a richer character-aware variant that keeps the original minimal
prompt intact for comparison:

```text
test/prompts/multimodal_action_summary_system_prompt_with_characters.txt
```

For a more narrative, translator-oriented version that pushes the model toward
actions, interactions, and transformations, use:

```text
test/prompts/multimodal_action_summary_system_prompt_narrative_focus.txt
```

You can edit that file directly, or point to another prompt file:

```bash
.venv/bin/python test/multimodal_action_summary.py \
  --spread-pages 10,11 \
  --models gpt-4o,gpt-5.5 \
  --system-prompt-file path/to/your_prompt.txt
```

Example with the richer Barbapapa family context:

```bash
.venv/bin/python test/multimodal_action_summary.py \
  --spread-pages 10,11 \
  --models gpt-4o,gpt-5.5 \
  --system-prompt-file test/prompts/multimodal_action_summary_system_prompt_with_characters.txt
```

Example with the narrative-focused prompt:

```bash
.venv/bin/python test/multimodal_action_summary.py \
  --spread-pages 10,11 \
  --models gpt-4o,gpt-5.5 \
  --system-prompt-file test/prompts/multimodal_action_summary_system_prompt_narrative_focus.txt
```

If you already have a rendered spread image and want to test that exact file,
use `--image` instead of `--spread-pages`:

```bash
.venv/bin/python test/multimodal_action_summary.py \
  --image translation/_multimodal_debug/20260601_135443/segment_02_pages_10-11.png \
  --models gpt-4o,gpt-5.5 \
  --detail high
```

Useful knobs for this evaluation:

- `--detail low|high` changes the OpenAI image-detail setting without changing the image file
- `--dpi 110` changes the rendered spread resolution when using `--spread-pages`
- `--user-prompt "..."` lets you tighten or relax the output instruction. By default, the script asks only: `Summarize the visible action in this double-page spread.`
- `--output path/to/report.md` writes the comparison report to a specific location
- `--no-cache` bypasses the local response cache for a fresh rerun

## Try Single-Page French Preprocessing

The standalone preprocessing experiment adapts the original French text from
double-page composition to page-by-page digital reading. It is not integrated
into the translation pipeline.

For each textless illustrated spread, it first asks the model for a conservative
editorial plan, then writes exactly one non-empty French text block per physical
body page. This lets it split or move existing text and add short transitions,
descriptions, or suspense where a formerly silent page would otherwise read
awkwardly.

Run a small two-spread trial first:

```bash
.venv/bin/python -m src.translation.preprocessing \
  --model openai:gpt-4o \
  --max-spreads 2 \
  --save-images \
  --render-pdf
```

Try the story-aware variant on the same first two spreads:

```bash
.venv/bin/python -m src.translation.preprocessing \
  --mode story \
  --model openai:gpt-4o \
  --max-spreads 2 \
  --story-spreads-per-chunk 5 \
  --save-images \
  --render-pdf \
  --no-cache
```

Run the complete book:

```bash
.venv/bin/python -m src.translation.preprocessing \
  --model openai:gpt-5.5
```

Run the complete book in story mode, processing five double-page spreads per
model call:

```bash
.venv/bin/python -m src.translation.preprocessing \
  --mode story \
  --model openai:gpt-5.5 \
  --story-spreads-per-chunk 5 \
  --render-pdf
```

Generate and inspect only the page-level story plan for the Barbapapa source,
without writing adapted page text:

```bash
.venv/bin/python -m src.translation.preprocessing \
  --mode story \
  --story-planner only \
  --source l_arbre_de_barbapapa_INT.repaired.txt \
  --pdf flag_ship__l_arbre_de_barbapapa_INT.pdf \
  --model openai:gpt-5.5 \
  --skip-first 5 \
  --skip-last 4 \
  --save-images
```

Planner-only mode writes `<source>.story_plan.json` and
`preprocessing_report.json` under a new timestamped preprocessing directory.
It uses the normal response cache. Prompt or image changes create a new cache
key automatically, while an identical rerun reuses the cached plan.

By default, output is written under
`translation/_preprocessing/<timestamp>/`. The directory contains the final
page-aligned `.txt` file and `preprocessing_report.json`, which records the
editorial plan, intervention, and result for each spread. Useful options:

- `--source path/to/cleaned_french.txt` and `--pdf path/to/book.pdf`
- `--mode spread` for the local spread-by-spread workflow, or `--mode story`
  for sequential story-aware chunks with bounded image context
- `--model openai:gpt-5.5`, `anthropic:<model>`, or `gemini:<model>`
- `--temperature 0.8` to control creative variation on models that support it
- `--skip-first 5 --skip-last 4` to select the PDF body-page window
- `--output path/to/result.txt` and `--artifacts-dir path/to/artifacts`
- `--max-spreads N` for inexpensive prompt trials
- `--story-spreads-per-chunk 5` to opt into chunking and control how many
  double-page spreads are sent in each story-mode call; when omitted, story
  mode keeps the original single-call behavior
- `--story-planner on` to generate a page-level visual plan before writing;
  use `--story-planner only` to inspect the plan without generating page text
- `--planner-output path/to/plan.json` to choose where planner-only output is saved
- `--allow-preprocessed-source` to intentionally preprocess a generated
  `.single_page` input; these inputs are rejected by default to prevent
  accidental second-generation drift
- `--save-images` to retain the exact textless spread images used by the model
- `--no-cache` to force fresh model responses while iterating on prompts
- `--render-pdf` to create a PDF preview containing only generated pages
- `--pdf-output path/to/preview.pdf` to choose the rendered PDF path
- `--render-full-pdf` to keep all original PDF pages in the preview
- `--font-file path/to/font.ttf` if the PDF preview needs an embedded font

Completed spreads are saved incrementally, so the partial text and report remain
available if a later model response fails validation.

To render an existing preprocessing text file without making any model calls:

```bash
.venv/bin/python -m src.translation.preprocessing \
  --render-from-text translation/_preprocessing/<timestamp>/l_arbre_de_barbapapa_INT.repaired.single_page.txt \
  --pdf-output translation/_preprocessing/<timestamp>/single_page_preview.pdf
```

For a one-spread trial, the rendered PDF contains only the generated pages,
starting from page 6. For a two-spread trial, the preview PDF contains pages
6-9 only; unprocessed spreads are not stored in that preview.

`--mode story` partitions the original French text into explicitly page-labelled
sections. Only the current chunk's images are sent. Each later chunk also
receives the two previous validated page texts for continuity. With
`--story-planner on`, writer calls receive the global story arc, one preceding
and two following source spreads, plus adjacent planner pages marked as
context-only. This keeps the current images prominent without losing the chunk
handoff. Planner-off runs retain the complete before/after source context for
comparison. `--max-spreads` limits the total run size, while
`--story-spreads-per-chunk` controls the image context of each model call. If
the latter is omitted, all selected spreads are processed in one call as before.

## Workflow Walkthrough

The runnable script is `main.py`. It loads configuration from environment
variables, builds the Burr application from `src/translation/workflow.py`, runs
the workflow, saves artifacts under `translation/`, and prints the results.

The workflow has the same high-level stages in both modes:

1. `generate_candidates`
2. `critique_candidates`
3. `summarize_critic`
4. `generate_final_text`

### Text Mode

`WORKFLOW_MODE=text` is the default.

In this mode:

- the full repaired French text is sent as one unit
- each active candidate generates one full-book translation
- OpenAI candidates use the translation prompt plus glossary guidance
- external baselines such as Google Translate are called as text-only candidates
- the critic compares the candidate full translations paragraph by paragraph
- the critic summary is then passed to the final synthesizer
- the final synthesizer produces one coherent final translation for the whole book

So text mode is:

- one candidate call per candidate per language
- one critic call per language
- one critic-summary call per language
- one final-synthesis call per language

### Multimodal Mode

`WORKFLOW_MODE=multimodal` switches candidate generation to spread-level
translation while keeping the critic and final synthesizer text-only.

In this mode:

- the cleaned French text is first aligned to non-empty body pages in the source PDF
- consecutive body pages are grouped by double-page spread
- for each spread, the workflow renders one textless spread image from the source PDF
- each candidate translates one spread at a time instead of the whole book at once
- the prompt for a spread includes:
  - the current spread French text
  - glossary guidance
  - the previous translated target-language spread history
  - optional previous raw French context depending on `SOURCE_CONTEXT_WINDOW`
  - the current textless spread image for multimodal-capable providers
- if both pages of a spread contain text, they are translated together in one call and returned in page order, separated by a blank line
- after all spread translations are generated, they are stitched back together into one candidate text per language
- from that point on, the critic, critic summary, and final synthesizer work the same way as in text mode

So multimodal mode is:

- multiple spread-level candidate calls per candidate per language
- then the same critic/summary/final steps as text mode

### Caching And Recovery

Both modes use the same recovery mechanisms:

- LLM and external translation responses are cached under `.translation_cache/`
- repeated runs with the same inputs reuse cached responses when possible
- transient LiteLLM provider failures are retried automatically
- partial artifacts are written to `translation/{language_code}/{run_id}/` after completed workflow steps, so candidate outputs survive later failures

## Generate A Translated PDF

After producing a translated text file, you can build a translated PDF with
`src/utils/pdf_translation_overlay.py`.

The script assumes:

- the first 5 pages and last 4 pages are skipped
- each non-empty body page corresponds to one paragraph in the translation file
- the original PDF contains selectable text that PyMuPDF can read

Example for Finnish:

```bash
python src/utils/pdf_translation_overlay.py \
  flag_ship__l_arbre_de_barbapapa_INT.pdf \
  translation/l_arbre_de_barbapapa_INT_fi.txt \
  -o output_fi.pdf
```

This script:

- finds non-empty pages between the skipped front and back matter
- redacts the original text on those pages
- inserts the translated paragraph into the detected text area

### Replace Barbapapa Names On Specific Pages

Some early pages such as pages 2 and 3 may contain standalone character names
outside the main body-page translation range. You can replace those exact French
names with the glossary CSV:

```bash
python src/utils/pdf_translation_overlay.py \
  flag_ship__l_arbre_de_barbapapa_INT.pdf \
  translation/l_arbre_de_barbapapa_INT_fi.txt \
  -o output_fi.pdf \
  --glossary-csv "Noms barbapapas - Sheet1.csv" \
  --glossary-language Finnish \
  --glossary-pages 2,3
```

This optional pass uses exact name mappings from `French` to the target language
column in the CSV, for example `Barbabelle -> Barbapupu` and
`François -> Kari`.

### Fonts For Hindi And Tamil

For scripts outside basic Latin, pass a Unicode font file so the inserted text
renders correctly:

```bash
python src/utils/pdf_translation_overlay.py \
  flag_ship__l_arbre_de_barbapapa_INT.pdf \
  translation/l_arbre_de_barbapapa_INT_hi.txt \
  -o output_hi.pdf \
  --font-file /path/to/NotoSansDevanagari-Regular.ttf
```

Example for Tamil:

```bash
python src/utils/pdf_translation_overlay.py \
  flag_ship__l_arbre_de_barbapapa_INT.pdf \
  translation/l_arbre_de_barbapapa_INT_ta.txt \
  -o output_ta.pdf \
  --font-file /path/to/NotoSansTamil-Regular.ttf
```

If a translation is longer than the original text, the script will try smaller
font sizes automatically. If it still does not fit, it raises an error so you
can inspect that page manually.

## Compare Any Two Text Files

To generate a Markdown report with a ROUGE-L score and a side-by-side diff for
any two text files, run:

```bash
.venv/bin/python src/utils/text_comparison_report.py \
  path/to/left.txt \
  path/to/right.txt \
  -o comparison_report.md \
  --left-label left_version \
  --right-label right_version \
  --title "Left vs Right"
```

## Access The Burr Logs

The workflow writes Burr traces under:

```text
/private/tmp/.burr/children-book-translation-advanced
```

To inspect the logs in the Burr UI from the project root, run:

```bash
burr_path=/private/tmp/.burr .venv/bin/burr
```

This command has two parts on one line:

- `burr_path=/private/tmp/.burr` tells the Burr UI where to read tracked runs
- `.venv/bin/burr` launches the Burr UI from this project's virtual environment

Do not run just:

```bash
/private/tmp/.burr burr
```

That tries to execute the directory itself and leads to `zsh: permission denied`.

## Other Workflow Demos

- `test/langraph_simple_workflow.py` uses LangGraph.
- `test/microsoft_agent_workflow.py` uses Microsoft Agent Framework workflows.
- `test/burr_workflow.py` uses a simple Burr example.
- `test/crewai_workflow.py` uses CrewAI.
