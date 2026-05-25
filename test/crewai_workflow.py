# pip install -U crewai python-dotenv rich

import os
from pathlib import Path

from dotenv import load_dotenv
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

# CrewAI initializes local storage and telemetry support during import. Keep both
# inside this repository so the example stays self-contained.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HOME", str(PROJECT_ROOT))
os.environ.setdefault("CREWAI_STORAGE_DIR", str(PROJECT_ROOT / ".crewai"))
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Agent, Crew, LLM, Process, Task

load_dotenv()

console = Console()


input_text = """
Milo the little mouse found a shiny red button under the old oak tree.
"Maybe it belongs to the moon!" he whispered.
"""


def hindi_prompt(text: str) -> str:
    return f"""
Translate the following children's book text into natural Hindi.

Style:
- For children aged 5-8
- Warm, simple, playful
- Avoid overly formal or Sanskrit-heavy Hindi
- Keep it easy to read aloud

Text:
{text}
"""


def tamil_prompt(text: str) -> str:
    return f"""
Translate the following children's book text into natural Tamil.

Style:
- For children aged 5-8
- Warm, simple, playful
- Prefer natural modern Tamil
- Avoid overly formal literary Tamil
- Keep it easy to read aloud

Text:
{text}
"""


llm = LLM(model="gpt-4o", temperature=0.3)

# CrewAI's basic shape is: Agent(s) do Task(s), and a Crew coordinates them.
translator = Agent(
    role="Children's book translator",
    goal="Translate short children's stories in a warm, simple style.",
    backstory="You translate picture-book text for children aged 5-8.",
    llm=llm,
    verbose=False,
    allow_delegation=False,
)

hindi_task = Task(
    description=hindi_prompt(input_text),
    expected_output="Only the Hindi translation.",
    agent=translator,
)

tamil_task = Task(
    description=tamil_prompt(input_text),
    expected_output="Only the Tamil translation.",
    agent=translator,
)

crew = Crew(
    agents=[translator],
    tasks=[hindi_task, tamil_task],
    process=Process.sequential,
    memory=False,
    verbose=False,
)


if __name__ == "__main__":
    result = crew.kickoff()
    task_outputs = result.tasks_output

    console.print("\n[bold]Original text[/bold]")
    console.print(Panel(input_text.strip(), title="English", border_style="blue"))

    console.print("\n[bold]Translations[/bold]")
    console.print(
        Columns(
            [
                Panel(str(task_outputs[0].raw), title="Hindi", border_style="green"),
                Panel(str(task_outputs[1].raw), title="Tamil", border_style="magenta"),
            ],
            equal=True,
            expand=True,
        )
    )
