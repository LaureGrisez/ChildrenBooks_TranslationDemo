# Children Books Translation Demo

Small demo that translates a short children's story from English into Hindi and
Tamil using OpenAI.

There are two versions so you can compare the implementation style:

- `test/langraph_simple_workflow.py` uses LangGraph.
- `test/microsoft_agent_workflow.py` uses Microsoft Agent Framework workflows.
- `test/burr_workflow.py` uses Burr.
- `test/crewai_workflow.py` uses CrewAI.

## Run The LangGraph Example

From the project root, run:

```bash
.venv/bin/python test/langraph_simple_workflow.py
```

## Run The Microsoft Agent Framework Example

From the project root, run:

```bash
.venv/bin/python test/microsoft_agent_workflow.py
```

## Run The Burr Example

From the project root, run:

```bash
.venv/bin/python test/burr_workflow.py
```

## Run The CrewAI Example

From the project root, run:

```bash
.venv/bin/python test/crewai_workflow.py
```

The script uses this sample story:

```text
Milo the little mouse found a shiny red button under the old oak tree.
"Maybe it belongs to the moon!" he whispered.
```

It prints the original English text plus Hindi and Tamil translations.

## Requirements

- Python 3.10 or later
- A local virtual environment at `.venv`
- Dependencies installed in that environment:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -U pip langgraph langchain-openai agent-framework-openai burr crewai python-dotenv rich
```

- A `.env` file containing:

```text
OPENAI_API_KEY=your_api_key_here
```
