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
.venv/bin/python -m pip install -U pip langgraph langchain-openai agent-framework-openai burr crewai python-dotenv rich deepl google-cloud-translate pymupdf
```

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

Notes:

- Replace `Finnish` with any language present in `Noms barbapapas - Sheet1.csv`.
- If `TARGET_LANGUAGES` is omitted, the workflow uses all CSV languages except `French`.
- Candidate generation follows this default incremental order: `google_translation`, `gpt4o`, `gpt5_5`, `claude_sonnet_4_6`, `gemini_3`.
- `MAX_PARALLEL_CANDIDATES` limits how many candidates from that order are used.
- `CANDIDATE_NAMES` lets you choose the exact candidate set explicitly.
- More than 3 active candidates is deprecated and not supported yet. For now, keep `MAX_PARALLEL_CANDIDATES<=3` and choose at most 3 names in `CANDIDATE_NAMES`.
- The default source text is `l_arbre_de_barbapapa_INT.repaired.txt`.
- Final translations are also exported as plain text under `translation/`, for example `translation/l_arbre_de_barbapapa_INT_fi.txt`.
- Each run also writes versioned artifacts under `translation/{language_code}/{run_id}/`, including `candidates/`, the final text, and a Markdown comparison report.
- OpenAI and external translation responses are cached under `.translation_cache/`, so rerunning the same job can recover from transient failures without recomputing earlier successful steps.
- OpenAI calls retry transient connection/server errors automatically. You can tune this with `OPENAI_RETRY_ATTEMPTS` and `OPENAI_RETRY_BASE_DELAY_SECONDS`.
- `WORKFLOW_MODE=text` is the default. `WORKFLOW_MODE=multimodal` switches candidate generation to spread-aligned translation using the source PDF plus local source-text context.
- In the current multimodal version, candidate generation uses the textless spread images, while the critic and final synthesizer remain text-only.
- In multimodal mode, each new segment also receives the previously translated target-language segment history. `TARGET_HISTORY_WINDOW=1` is the default.
- Multimodal mode groups consecutive body pages by double-page spread. When both pages in a spread contain text, they are translated together in one multimodal call and returned in page order.
- Multimodal mode assumes the cleaned French text has one paragraph per non-empty body page in the source PDF. For `l_arbre_de_barbapapa`, that mapping is currently valid and currently groups into 16 spread calls.
- `SOURCE_CONTEXT_WINDOW=0` is now the default in multimodal mode, so the current French paragraph stays primary. You can raise it if you want some previous raw French context as well.
- You can tune multimodal context with `TARGET_HISTORY_WINDOW` and `SOURCE_CONTEXT_WINDOW`, and spread rendering size with `MULTIMODAL_IMAGE_DPI`.
- The runnable app entrypoint is `main.py`; the translation code itself lives under `src/translation/`.

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

- OpenAI and external translation responses are cached under `.translation_cache/`
- repeated runs with the same inputs reuse cached responses when possible
- transient OpenAI failures are retried automatically
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
