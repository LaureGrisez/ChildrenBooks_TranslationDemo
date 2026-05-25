# pip install -U langgraph langchain-openai rich

import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

load_dotenv()

console = Console()

class TranslationState(TypedDict):
    text: str
    hindi: str
    tamil: str

model = ChatOpenAI(
    model="gpt-4o",  # or "gpt-5.5" / your preferred OpenAI API model
    temperature=0.3,
)

def translate_hindi(state: TranslationState):
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
    return {"hindi": response.content}

def translate_tamil(state: TranslationState):
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
    return {"tamil": response.content}

graph_builder = StateGraph(TranslationState)

graph_builder.add_node("translate_hindi", translate_hindi)
graph_builder.add_node("translate_tamil", translate_tamil)

# Run Hindi and Tamil translation branches from the same input
graph_builder.add_edge(START, "translate_hindi")
graph_builder.add_edge(START, "translate_tamil")

graph_builder.add_edge("translate_hindi", END)
graph_builder.add_edge("translate_tamil", END)

graph = graph_builder.compile()

input_text = """
Milo the little mouse found a shiny red button under the old oak tree.
"Maybe it belongs to the moon!" he whispered.
"""

result = graph.invoke({"text": input_text})

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
