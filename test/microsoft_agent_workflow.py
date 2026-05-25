# pip install -U agent-framework-openai rich
#
# This script demonstrates a small Microsoft Agent Framework workflow:
# 1. Take one English children's-book passage.
# 2. Fan it out into two translation requests, one for Hindi and one for Tamil.
# 3. Send each request to a separate OpenAI-backed agent.
# 4. Collect both translations and print them side by side.

import asyncio
import warnings
from dataclasses import dataclass

# Hide noisy framework warnings so the Rich output stays readable.
warnings.filterwarnings("ignore", message=r"\[HARNESS\].*")
warnings.filterwarnings("ignore", message=r"\[SKILLS\].*")

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    executor,
)
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Never

# Loads environment variables from a local .env file.
# The OpenAI client expects credentials such as OPENAI_API_KEY to be available.
load_dotenv()

# Rich console used for nicely formatted terminal output.
console = Console()


@dataclass
class TranslationResult:
    """Typed workflow output for one completed translation."""

    language: str
    text: str


# The source text that will be translated by both agents.
input_text = """
Milo the little mouse found a shiny red button under the old oak tree.
"Maybe it belongs to the moon!" he whispered.
"""


def hindi_prompt(text: str) -> str:
    """Build the user prompt sent to the Hindi translation agent."""

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
    """Build the user prompt sent to the Tamil translation agent."""

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


@executor(id="start", output=str)
async def start(text: str, ctx: WorkflowContext[str]) -> None:
    """First workflow node: forwards the raw input text into the graph."""

    await ctx.send_message(text)


@executor(id="make_hindi_request", output=AgentExecutorRequest)
async def make_hindi_request(text: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    """Convert the shared input text into an agent request for Hindi."""

    request = AgentExecutorRequest(
        # AgentExecutor expects a chat-style request: role + message contents.
        messages=[Message(role="user", contents=[hindi_prompt(text)])],
        # True means the agent should produce a final response for this request.
        should_respond=True,
    )
    await ctx.send_message(request)


@executor(id="make_tamil_request", output=AgentExecutorRequest)
async def make_tamil_request(text: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    """Convert the shared input text into an agent request for Tamil."""

    request = AgentExecutorRequest(
        messages=[Message(role="user", contents=[tamil_prompt(text)])],
        should_respond=True,
    )
    await ctx.send_message(request)


@executor(id="collect_hindi", workflow_output=TranslationResult)
async def collect_hindi(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[Never, TranslationResult],
) -> None:
    """Turn the Hindi agent response into a typed workflow output."""

    await ctx.yield_output(
        TranslationResult(language="Hindi", text=response.agent_response.text)
    )


@executor(id="collect_tamil", workflow_output=TranslationResult)
async def collect_tamil(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[Never, TranslationResult],
) -> None:
    """Turn the Tamil agent response into a typed workflow output."""

    await ctx.yield_output(
        TranslationResult(language="Tamil", text=response.agent_response.text)
    )


async def main() -> None:
    """Create agents, wire the workflow graph, run it, and print results."""

    # Shared chat client used by both translation agents.
    chat_client = OpenAIChatCompletionClient(model="gpt-4o")

    # AgentExecutor adapts an OpenAI chat agent so it can be used as a workflow node.
    hindi_agent = AgentExecutor(
        chat_client.as_agent(
            name="HindiTranslator",
            # Keep the agent output clean: no explanations, only the translation.
            instructions="Return only the Hindi translation.",
            default_options={"temperature": 0.3},
        ),
        id="hindi_agent",
    )
    tamil_agent = AgentExecutor(
        chat_client.as_agent(
            name="TamilTranslator",
            instructions="Return only the Tamil translation.",
            # Low temperature makes the translations more consistent between runs.
            default_options={"temperature": 0.3},
        ),
        id="tamil_agent",
    )

    # Build the directed workflow graph:
    # start
    #   -> make_hindi_request -> hindi_agent -> collect_hindi
    #   -> make_tamil_request -> tamil_agent -> collect_tamil
    workflow = (
        WorkflowBuilder(start_executor=start, output_from=[collect_hindi, collect_tamil])
        # Fan out sends the same start output to both request-building nodes.
        .add_fan_out_edges(start, [make_hindi_request, make_tamil_request])
        .add_edge(make_hindi_request, hindi_agent)
        .add_edge(hindi_agent, collect_hindi)
        .add_edge(make_tamil_request, tamil_agent)
        .add_edge(tamil_agent, collect_tamil)
        .build()
    )

    # Run the graph and collect any values produced by ctx.yield_output(...).
    events = await workflow.run(input_text)
    translations = {
        result.language: result.text for result in events.get_outputs()
    }

    # Display the original text and both translations in terminal panels.
    console.print("\n[bold]Original text[/bold]")
    console.print(Panel(input_text.strip(), title="English", border_style="blue"))

    console.print("\n[bold]Translations[/bold]")
    console.print(
        Columns(
            [
                Panel(translations.get("Hindi", ""), title="Hindi", border_style="green"),
                Panel(translations.get("Tamil", ""), title="Tamil", border_style="magenta"),
            ],
            equal=True,
            expand=True,
        )
    )


if __name__ == "__main__":
    # Entry point for running this file directly: python test/microsoft_agent_workflow.py
    asyncio.run(main())
