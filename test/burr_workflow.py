# pip install -U burr openai python-dotenv rich
#
# This script demonstrates a small Burr workflow:
# 1. Store the English children's-book passage in Burr state.
# 2. Run a Hindi translation action.
# 3. Run a Tamil translation action after Hindi completes.
# 4. Print the original text and translations side by side.

from openai import OpenAI
from dotenv import load_dotenv
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

from burr.core import ApplicationBuilder, State, action

# Loads environment variables from a local .env file.
# The OpenAI client expects credentials such as OPENAI_API_KEY to be available.
load_dotenv()

# Rich console used for nicely formatted terminal output.
console = Console()

# OpenAI client shared by both Burr actions.
client = OpenAI()


# The source text that will be stored in Burr state and translated.
input_text = """
Milo the little mouse found a shiny red button under the old oak tree.
"Maybe it belongs to the moon!" he whispered.
"""


def hindi_prompt(text: str) -> str:
    """Build the prompt used for the Hindi translation call."""

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
    """Build the prompt used for the Tamil translation call."""

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
    """Send one prompt to OpenAI and return the assistant's text response."""

    response = client.chat.completions.create(
        model="gpt-4o",
        # Low temperature makes the translations more consistent between runs.
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# Burr actions read from and write to State. This keeps the workflow explicit:
# each action updates the shared translation state, then the graph moves on.
@action(reads=["text"], writes=["hindi"])
def translate_hindi(state: State) -> State:
    """Read state['text'], translate it into Hindi, and write state['hindi']."""

    return state.update(hindi=ask_openai(hindi_prompt(state["text"])))


@action(reads=["text"], writes=["tamil"])
def translate_tamil(state: State) -> State:
    """Read state['text'], translate it into Tamil, and write state['tamil']."""

    return state.update(tamil=ask_openai(tamil_prompt(state["text"])))


def build_application():
    """Build the Burr application graph used by this example."""

    return (
        ApplicationBuilder()
        # Initial state available to all actions.
        .with_state(text=input_text)
        # Register both action functions as nodes in the Burr application.
        .with_actions(translate_hindi, translate_tamil)
        # After translate_hindi completes, Burr should run translate_tamil.
        .with_transitions(("translate_hindi", "translate_tamil"))
        # Start the application at the Hindi translation action.
        .with_entrypoint("translate_hindi")
        .build()
    )


if __name__ == "__main__":
    # Build and run the application until the Tamil action has completed.
    app = build_application()
    _, _, state = app.run(halt_after=["translate_tamil"])

    # Display the original text and both translations in terminal panels.
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
