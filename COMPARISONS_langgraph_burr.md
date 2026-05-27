# COMPARISONS

The important point in this comparison is that the workflow behavior is now intentionally aligned between the two scripts: same candidate generation strategy, same critic contract, same final synthesis logic, and very similar terminal outputs. What remains are mostly framework differences rather than workflow-design differences.

## Assembling the app

### Langgraph
```py
def build_graph():
    graph_builder = StateGraph(TranslationState)
    graph_builder.add_node("generate_candidates", generate_candidates)
    graph_builder.add_node("critique_candidates", critique_candidates)
    graph_builder.add_node("summarize_critic", summarize_critic)
    graph_builder.add_node("generate_final_text", generate_final_text)

    graph_builder.add_edge(START, "generate_candidates")
    graph_builder.add_edge("generate_candidates", "critique_candidates")
    graph_builder.add_edge("critique_candidates", "summarize_critic")
    graph_builder.add_edge("summarize_critic", "generate_final_text")
    graph_builder.add_edge("generate_final_text", END)

    return graph_builder.compile()
```

LangGraph feels more like explicitly building a graph object step by step: declare nodes, add edges, then compile it.

### Burr
```python
def build_application():
    tracker = LocalTrackingClient(project=PROJECT_NAME, storage_dir=BURR_STORAGE_DIR)
    return (
        ApplicationBuilder()
        .with_state(
            text=input_text,
            target_languages=TARGET_LANGUAGES,
            decision_log=[],
            critic_reasoning={},
            critic_winners={},
        )
        .with_actions(
            generate_candidates,
            critique_candidates,
            summarize_critic,
            generate_final_text,
        )
        .with_transitions(
            ("generate_candidates", "critique_candidates"),
            ("critique_candidates", "summarize_critic"),
            ("summarize_critic", "generate_final_text"),
        )
        .with_entrypoint("generate_candidates")
        .with_tracker(tracker)
        .build()
    )
```

Burr feels more like building an application pipeline: register actions, declare transitions, set the entrypoint, then optionally attach tracking.

## State Declaration

### Langgraph

```python
class TranslationState(TypedDict, total=False):
    text: str
    target_languages: list[str]
    decision_log: list[dict[str, Any]]
    candidate_translations: dict[str, list[dict[str, Any]]]
    critic_reviews: dict[str, str]
    critic_reasoning: dict[str, str]
    critic_winners: dict[str, str]
    critic_summaries: dict[str, str]
    final_translations: dict[str, str]
```

### Burr
```python
@action(
    reads=["text", "candidate_translations", "decision_log"],
    writes=["critic_reviews", "critic_reasoning", "critic_winners", "decision_log"],
)
def critique_candidates(state: State) -> State:
    ...

    return updated.update(
        critic_reviews=reviews,
        critic_reasoning=reasoning,
        critic_winners=winners,
    )
```

Burr does not declare the whole state schema in one TypedDict. Instead:
- some initial state is declared in `.with_state(...)`
- each action declares the parts of state it reads and writes through `@action(reads=..., writes=...)`

So LangGraph is more centralized and typed at the state-definition level, while Burr is more incremental and action-oriented.

## Block formulations

### Langgraph
```python
def generate_candidates(state: TranslationState) -> dict[str, Any]:
    # ...
    return {"candidate_translations": all_candidates, "decision_log": events}
```

Nodes return partial state updates as dictionaries, which LangGraph merges into the current state.

### Burr
```python
def generate_candidates(state: State) -> State:
    # ...
    return updated.update(candidate_translations=all_candidates)
```

Actions receive a Burr `State` object and explicitly return a new updated `State`.

## Tracing

### Langgraph
```python
@traceable(
    run_type="chain",
    name="langgraph_summarize_critic",
    project_name=LANGSMITH_PROJECT,
    process_inputs=serialize_for_trace,
    process_outputs=serialize_for_trace,
)
```
Access though : [https://eu.smith.langchain.com]

Tracing is more explicit and programmable here: we added decorators around nodes and helper functions, and also traced the external Google translation call.

### Burr
```python
tracker = LocalTrackingClient(project=PROJECT_NAME, storage_dir=BURR_STORAGE_DIR)
```
and 
`.with_tracker(tracker)` when building the app

Tracking is more built into the application runtime itself: once attached to the app, Burr records the action execution flow locally for the Burr UI.

## Run the app

### Langgraph

Here to get the intermediate logs printed in the terminal :
```python
graph = build_graph()
initial_state: TranslationState = {
    "text": input_text,
    "target_languages": TARGET_LANGUAGES,
    "decision_log": [],
}

config = {
    "run_name": "children_book_translation_advanced",
    "tags": ["children-book", "translation", "adversarial-candidates"],
    "metadata": {
        "project": LANGSMITH_PROJECT,
        "languages": TARGET_LANGUAGES,
        "external_translator": EXTERNAL_TRANSLATOR,
        "max_parallel_candidates": MAX_PARALLEL_CANDIDATES,
    },
}

result: TranslationState = initial_state

for update in graph.stream(initial_state, config=config):
    print_stream_update(update)
    node_update = next(iter(update.values()))
    result = {**result, **node_update}

print_results(result)
```

else :
```python
graph = build_graph()
result = graph.invoke(initial_state, config=config)
```

### Burr
```python
if __name__ == "__main__":
    app = build_application()
    _, _, final_state = app.run(halt_after=["generate_final_text"])
    print_results(final_state)
```

LangGraph gives two useful execution styles very naturally:
- `graph.invoke(...)` for one-shot execution
- `graph.stream(...)` for incremental updates

Burr is more centered on the application runner and step progression through `app.run(...)`, with tracking attached at the app level.

## Short Summary

- LangGraph feels closer to a general graph orchestration library with flexible tracing through LangSmith.
- Burr feels closer to a state-machine application framework with first-class local tracking and explicit state transitions.
- LangGraph emphasizes graph compilation and partial state updates.
- Burr emphasizes actions, read/write contracts, and runtime-managed state.
- Once the workflow behavior is aligned, the choice is mostly about ergonomics, observability style, and how explicit you want state transitions to be.

