"""App entrypoint for the Burr translation workflow."""

from __future__ import annotations

from .config import TranslationWorkflowConfig
from .workflow import build_application, print_results, save_final_translations


def run() -> None:
    """Build, run, persist, and display one translation workflow execution."""

    app, config = build_application(TranslationWorkflowConfig.from_env())
    terminal_action = (
        "repair_flagged_paragraphs" if config.is_panel_mode() else "generate_final_text"
    )
    _, _, final_state = app.run(halt_after=[terminal_action])
    saved_paths = save_final_translations(final_state, config)
    print_results(final_state, config, saved_paths=saved_paths)


def main() -> None:
    """CLI entrypoint."""

    run()


if __name__ == "__main__":
    main()
