"""MemSlides - an agentic and reflective presentation generation system."""

__version__ = "1.0.0"

from memslides.contracts import (
    DeckRequest,
    DeckResult,
    MemoryOptions,
    RevisionRequest,
    SessionOptions,
    TemplateOptions,
)
from memslides.session import MemSlidesSession

__all__ = [
    "MemSlidesSession",
    "DeckRequest",
    "RevisionRequest",
    "DeckResult",
    "SessionOptions",
    "MemoryOptions",
    "TemplateOptions",
]
