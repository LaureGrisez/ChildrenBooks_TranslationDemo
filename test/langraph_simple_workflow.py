# pip install -U langgraph langchain-openai rich
#
# This script demonstrates a small LangGraph workflow:
# 1. Start with one English children's-book passage.
# 2. Run two translation nodes in parallel, one for Hindi and one for Tamil.
# 3. Store both translations in the graph state.
# 4. Print the original text and translations side by side.

import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

# Loads environment variables from a local .env file.
# ChatOpenAI expects credentials such as OPENAI_API_KEY to be available.
load_dotenv()

# Rich console used for nicely formatted terminal output.
console = Console()


class TranslationState(TypedDict):
    """State shared between all LangGraph nodes."""

    # Original source text that both translation nodes read.
    text: str
    # Hindi translation added by the translate_hindi node.
    hindi: str
    # Tamil translation added by the translate_tamil node.
    tamil: str


# Shared OpenAI chat model used by both translation nodes.
model = ChatOpenAI(
    model="gpt-4o",  # or "gpt-5.5" / your preferred OpenAI API model
    # Low temperature makes the translations more consistent between runs.
    temperature=0.3,
)


def translate_hindi(state: TranslationState):
    """LangGraph node that reads state['text'] and returns the Hindi translation."""

    prompt = f"""
Translate the following children's book text into natural Hindi.

Style:
- For children aged 5–8
- Warm, simple, playful
- Avoid overly formal or Sanskrit-heavy Hindi
- Keep it easy to read aloud

Text:
{state["text"]}
"""
    response = model.invoke(prompt)
    # Returning {"hindi": ...} updates only the hindi field in the graph state.
    return {"hindi": response.content}


def translate_tamil(state: TranslationState):
    """LangGraph node that reads state['text'] and returns the Tamil translation."""

    prompt = f"""
Translate the following children's book text into natural Tamil.

Style:
- For children aged 5–8
- Warm, simple, playful
- Prefer natural modern Tamil
- Avoid overly formal literary Tamil
- Keep it easy to read aloud

Text:
{state["text"]}
"""
    response = model.invoke(prompt)
    # Returning {"tamil": ...} updates only the tamil field in the graph state.
    return {"tamil": response.content}


# Create a graph builder whose state shape is defined by TranslationState.
graph_builder = StateGraph(TranslationState)

# Register the functions above as named graph nodes.
graph_builder.add_node("translate_hindi", translate_hindi)
graph_builder.add_node("translate_tamil", translate_tamil)

# Run Hindi and Tamil translation branches from the same input.
graph_builder.add_edge(START, "translate_hindi")
graph_builder.add_edge(START, "translate_tamil")

# Each translation branch can finish independently after updating the state.
graph_builder.add_edge("translate_hindi", END)
graph_builder.add_edge("translate_tamil", END)

# Compile the graph builder into an executable graph.
graph = graph_builder.compile()

# The source text that will be translated by both nodes.
input_text = """
Milo the little mouse found a shiny red button under the old oak tree.
"Maybe it belongs to the moon!" he whispered.
"""

# Run the graph with the initial state. LangGraph merges each node's returned
# dictionary into the final result.
result = graph.invoke({"text": input_text})

# Display the original text and both translations in terminal panels.
console.print("\n[bold]Original text[/bold]")
console.print(Panel(input_text.strip(), title="English", border_style="blue"))

console.print("\n[bold]Translations[/bold]")
console.print(
    Columns(
        [
            Panel(result["hindi"], title="Hindi", border_style="green"),
            Panel(result["tamil"], title="Tamil", border_style="magenta"),
        ],
        equal=True,
        expand=True,
    )
)
