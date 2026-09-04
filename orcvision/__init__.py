"""Orc-Vison — perception event stream + autonomous decision brain."""

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

__all__ = ["Detection", "PerceptionEvent", "__version__"]

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from orcvision.events import Detection, PerceptionEvent


def __getattr__(name: str) -> Any:
    """Lazily expose the perception schemas (PEP 562).

    ``orcvision.brain`` is pure standard library by design, but importing
    any submodule executes this package first. Loading the pydantic event
    schemas eagerly here would drag pydantic into every brain-only
    deployment — including edge targets that never touch the perception
    half. Deferring the import keeps ``import orcvision.brain`` dependency
    free while ``from orcvision import Detection`` still works unchanged.
    """
    if name in ("Detection", "PerceptionEvent"):
        from orcvision import events

        return getattr(events, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
