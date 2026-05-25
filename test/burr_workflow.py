# pip install -U burr openai python-dotenv rich

from openai import OpenAI
from dotenv import load_dotenv
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

from burr.core import ApplicationBuilder, State, action

load_dotenv()

console = Console()
client = OpenAI()


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


def ask_openai(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# Burr actions read from and write to State. This keeps the workflow explicit:
# each action updates the shared translation state, then the graph moves on.
@action(reads=["text"], writes=["hindi"])
def translate_hindi(state: State) -> State:
    return state.update(hindi=ask_openai(hindi_prompt(state["text"])))


@action(reads=["text"], writes=["tamil"])
def translate_tamil(state: State) -> State:
    return state.update(tamil=ask_openai(tamil_prompt(state["text"])))


def build_application():
    return (
        ApplicationBuilder()
        .with_state(text=input_text)
        .with_actions(translate_hindi, translate_tamil)
        .with_transitions(("translate_hindi", "translate_tamil"))
        .with_entrypoint("translate_hindi")
        .build()
    )


if __name__ == "__main__":
    app = build_application()
    _, _, state = app.run(halt_after=["translate_tamil"])

    console.print("\n[bold]Original text[/bold]")
    console.print(Panel(input_text.strip(), title="English", border_style="blue"))

    console.print("\n[bold]Translations[/bold]")
    console.print(
        Columns(
            [
                Panel(state["hindi"], title="Hindi", border_style="green"),
                Panel(state["tamil"], title="Tamil", border_style="magenta"),
            ],
            equal=True,
            expand=True,
        )
    )
