# pip install -U crewai python-dotenv rich
#
# This script demonstrates a small CrewAI workflow:
# 1. Create one translator agent.
# 2. Give that agent two translation tasks, one for Hindi and one for Tamil.
# 3. Let a Crew run the tasks sequentially.
# 4. Print the original text and translations side by side.

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

# Loads environment variables from a local .env file.
# The LLM expects credentials such as OPENAI_API_KEY to be available.
load_dotenv()

# Rich console used for nicely formatted terminal output.
console = Console()


# The source text that will be translated by both tasks.
input_text = """
Milo the little mouse found a shiny red button under the old oak tree.
"Maybe it belongs to the moon!" he whispered.
"""


def hindi_prompt(text: str) -> str:
    """Build the prompt used by the Hindi translation task."""

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
    """Build the prompt used by the Tamil translation task."""

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


# Shared language model used by the CrewAI agent.
llm = LLM(model="gpt-4o", temperature=0.3)

# CrewAI's basic shape is: Agent(s) do Task(s), and a Crew coordinates them.
translator = Agent(
    # The role, goal, and backstory tell CrewAI what kind of worker this is.
    role="Children's book translator",
    goal="Translate short children's stories in a warm, simple style.",
    backstory="You translate picture-book text for children aged 5-8.",
    llm=llm,
    verbose=False,
    # This example keeps all work inside this one agent.
    allow_delegation=False,
)

# First task: ask the translator agent for the Hindi version.
hindi_task = Task(
    description=hindi_prompt(input_text),
    expected_output="Only the Hindi translation.",
    agent=translator,
)

# Second task: ask the same translator agent for the Tamil version.
tamil_task = Task(
    description=tamil_prompt(input_text),
    expected_output="Only the Tamil translation.",
    agent=translator,
)

# The Crew coordinates which agents run which tasks and in what order.
crew = Crew(
    agents=[translator],
    tasks=[hindi_task, tamil_task],
    # Sequential means CrewAI runs the Hindi task, then the Tamil task.
    process=Process.sequential,
    # Memory is off so each script run stays simple and repeatable.
    memory=False,
    verbose=False,
)


if __name__ == "__main__":
    # kickoff() runs all tasks and returns a CrewOutput containing task results.
    result = crew.kickoff()
    task_outputs = result.tasks_output

    # The output list follows the same order as the tasks list above:
    # task_outputs[0] is Hindi and task_outputs[1] is Tamil.
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
