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

Notes:

- Replace `Finnish` with any language present in `Noms barbapapas - Sheet1.csv`.
- If `TARGET_LANGUAGES` is omitted, the workflow uses all CSV languages except `French`.
- The default source text is `l_arbre_de_barbapapa_INT.repaired.txt`.
- Final translations are also exported as plain text under `translation/`, for example `translation/l_arbre_de_barbapapa_INT_fi.txt`.
- Each run also writes versioned artifacts under `translation/{language_code}/{run_id}/`, including `candidates/`, the final text, and a Markdown comparison report.
- The runnable app entrypoint is `main.py`; the translation code itself lives under `src/translation/`.

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
