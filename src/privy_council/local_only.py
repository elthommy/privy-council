"""Local-only mode (M15): every Claude seat moves to a local Ollama model.

Personas are reseated up front, on copies, so routes, the `+` marks, recorded
models and estimates all follow; the registry's `local_only` guard refuses
anything that would still reach Claude.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from privy_council.config import OllamaModelConfig, Settings
from privy_council.models.persona import BackendKind, Persona


class LocalOnlyConfigError(ValueError):
    """The `[local_only]` section doesn't fit the loaded personas."""


def seat_model(settings: Settings, persona: Persona, moderators: Iterable[str]) -> str:
    """Return the local model a Claude persona runs on in local-only mode."""
    cfg = settings.local_only
    if persona.slug in moderators:
        return cfg.moderator_model
    return cfg.seats.get(persona.slug, cfg.default_model)


def local_personas(
    settings: Settings, personas: Mapping[str, Persona], moderators: Iterable[str]
) -> dict[str, Persona]:
    """Return copies of `personas` with every Claude seat on its local model."""
    check_seats(settings, personas)
    moderators = set(moderators)
    return {
        slug: (
            p.reseated(seat_model(settings, p, moderators), BackendKind.OLLAMA)
            if p.meta.backend is BackendKind.CLAUDE_CLI
            else p
        )
        for slug, p in personas.items()
    }


def check_seats(settings: Settings, personas: Mapping[str, Persona]) -> None:
    """Refuse a seat-map entry that names no persona, or a persona already local."""
    for slug in settings.local_only.seats:
        persona = personas.get(slug)
        if persona is None:
            raise LocalOnlyConfigError(f"[local_only.seats] {slug}: no such persona")
        if persona.meta.backend is not BackendKind.CLAUDE_CLI:
            raise LocalOnlyConfigError(
                f"[local_only.seats] {slug}: already a local seat ({persona.meta.model})"
            )


def local_settings(settings: Settings) -> Settings:
    """Return settings with the moderator model's num_ctx raised for the session.

    Every seat on that model gets it: Ollama reloads a model whose num_ctx changes.
    """
    cfg = settings.local_only
    models = dict(settings.ollama.models)
    current = models.get(cfg.moderator_model, OllamaModelConfig())
    models[cfg.moderator_model] = current.model_copy(update={"num_ctx": cfg.moderator_num_ctx})
    ollama = settings.ollama.model_copy(update={"models": models})
    return settings.model_copy(update={"ollama": ollama})


def apply_local_only(
    settings: Settings, personas: Mapping[str, Persona], moderators: Iterable[str]
) -> tuple[Settings, dict[str, Persona]]:
    """Return the settings and personas a local-only session runs with."""
    return local_settings(settings), local_personas(settings, personas, moderators)


def required_models(
    settings: Settings, personas: Mapping[str, Persona], moderators: Iterable[str]
) -> set[str]:
    """Return every Ollama model a local-only session may need (for `doctor`)."""
    seated = local_personas(settings, personas, moderators)
    return {p.meta.model for p in seated.values()} | {settings.local_only.moderator_model}
