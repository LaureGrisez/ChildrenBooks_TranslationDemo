"""Reusable translation workflow components."""

from .config import TranslationWorkflowConfig
from .glossary import CharacterGlossary, load_character_glossary

__all__ = [
    "CharacterGlossary",
    "TranslationWorkflowConfig",
    "build_application",
    "load_character_glossary",
    "print_results",
]


def __getattr__(name: str):
    """Load workflow entrypoints lazily so pure helpers remain credential-free."""

    if name in {"build_application", "print_results"}:
        from .workflow import build_application, print_results

        return {"build_application": build_application, "print_results": print_results}[name]
    raise AttributeError(name)
