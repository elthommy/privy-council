"""Load `config.toml` (+ optional gitignored `config.local.toml`) into Pydantic settings."""

from __future__ import annotations

import getpass
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

CONFIG_FILE = "config.toml"
LOCAL_CONFIG_FILE = "config.local.toml"
DEFAULT_USER_NAME = "User"


class ConfigError(Exception):
    """Raised when a config file can't be read or fails validation."""


class _Section(BaseModel):
    """Base for config sections: unknown keys are errors, to catch typos."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Section):
    """Directory locations; relative paths resolve against the config root."""

    personas: Path = Path("personas")
    councils: Path = Path("councils")
    data: Path = Path("data")


class ClaudeConfig(_Section):
    """Settings for the `claude` CLI backend."""

    binary: str = "claude"
    default_model: str = "sonnet"
    timeout_s: float = Field(default=180, gt=0)
    session_soft_cap_calls: int = Field(default=40, ge=0)
    daily_soft_cap_calls: int = Field(default=150, ge=0)
    # Warn when the real Pro quota (from the CLI's rate_limit_event) passes this.
    quota_warn_utilization: float = Field(default=0.8, ge=0.0, le=1.0)
    fallback_to_local: bool = False
    fallback_model: str = "qwen3:8b"


class OllamaModelConfig(_Section):
    """Per-model Ollama options."""

    num_ctx: int = Field(default=8192, gt=0)
    # qwen3 thinks by default; see docs/spike-notes.md. gpt-oss can't turn thinking
    # off, only take a level.
    think: bool | Literal["low", "medium", "high"] | None = None
    # Prompt tokens per batch (Ollama's default: 512). With most of a model in RAM,
    # each batch copies those layers to the GPU, so bigger batches read prompts
    # faster (spike 6). None leaves Ollama's default.
    num_batch: int | None = Field(default=None, gt=0)


class OllamaConfig(_Section):
    """Settings for the Ollama backend."""

    base_url: str = "http://localhost:11434"
    timeout_s: float = Field(default=300, gt=0)
    concurrency: int = Field(default=1, ge=1)
    keep_alive: str = "10m"
    models: dict[str, OllamaModelConfig] = Field(default_factory=dict)


class EngineConfig(_Section):
    """Debate engine settings."""

    context_last_k_messages: int = Field(default=6, ge=0)
    # Default hard cap per turn (personas may override); stops runaway answers only.
    max_output_tokens: int = Field(default=1500, gt=0)
    max_turn_chars_display: int = Field(default=2000, gt=0)
    # Floor turns per round asking members to check an error the moderator caught
    # (M11); 0 turns them off. Errors past the cap go to the synthesis.
    max_flaw_prompts_per_round: int = Field(default=2, ge=0)


class LocalSearchConfig(_Section):
    """Local web search through SearXNG, for local-only sessions."""

    # Opt-in and consent: with it on, local-only sessions send search queries to
    # SearXNG, which forwards them to web search engines.
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8888"
    max_queries: int = Field(default=4, ge=1, le=8)
    results_per_query: int = Field(default=5, ge=1, le=20)
    max_results: int = Field(default=12, ge=1, le=40)  # after merging the queries
    timeout_s: float = Field(default=20, gt=0)
    # Start SearXNG (scripts/searxng.sh) when a session needs it and it's down, and
    # stop it at exit if the session started it. Only a localhost base_url.
    manage_container: bool = True
    # Before round 1: one more search when the first note ends with "Missing: …" (the
    # Chair only learns the game, country or product from the first results).
    follow_up: bool = True


class ResearchConfig(_Section):
    """Web research by the moderator (claude CLI web tools); see docs/spike-notes.md."""

    enabled: bool = True  # false: /research is off unless `council new --research`
    # Searches the moderator may start on its own; past this it asks the user first.
    # The user's own `/research` is never capped (soft budget warnings still apply).
    auto_max_per_session: int = Field(default=3, ge=0)
    timeout_s: float = Field(default=300, gt=0)  # a call may run several searches
    max_output_tokens: int = Field(default=3000, gt=0)
    # Share of each member's prompt budget the facts may take (small num_ctx models).
    context_share: float = Field(default=0.25, gt=0.0, le=0.8)
    local: LocalSearchConfig = Field(default_factory=LocalSearchConfig)
    # Normal sessions: "claude" (its web search), "local" (SearXNG; the Chair still
    # writes the queries and the summary), or "auto" (Claude, and SearXNG whenever
    # Claude can't search). Local-only sessions always use SearXNG, whatever this says.
    provider: Literal["claude", "local", "auto"] = "claude"

    @model_validator(mode="after")
    def _local_needs_consent(self) -> ResearchConfig:
        """SearXNG sends queries off the PC: only with the [research.local] opt-in."""
        if self.provider != "claude" and not self.local.enabled:
            raise ValueError(
                f'provider = "{self.provider}" needs local search: set '
                "[research.local] enabled = true"
            )
        return self


class LocalOnlyConfig(_Section):
    """Local-only mode (M15): every Claude seat runs on a local Ollama model instead."""

    enabled: bool = False  # the default for `council new` / `compare new`
    default_model: str = Field(default="gpt-oss:120b", min_length=1)
    moderator_model: str = Field(default="nemotron-3.5-super:latest", min_length=1)
    # Applies to every seat on moderator_model: Ollama reloads a model whose num_ctx
    # changes, and Super is 90 GB. At 12k the trim cut facts the Chair needed (M15 bench).
    moderator_num_ctx: int = Field(default=24576, gt=0)
    seats: dict[str, str] = Field(default_factory=dict)  # persona slug -> Ollama model


class UserConfig(_Section):
    """How the council addresses the user."""

    # None: the login name, capitalized ("thomas" -> "Thomas"). Set it in
    # config.local.toml (gitignored) to keep a real name out of the repo.
    name: str | None = Field(default=None, min_length=1, max_length=32)


class MemoryConfig(_Section):
    """Persona memory settings."""

    private_cap_bytes: int = Field(default=4096, gt=0)
    shared_cap_bytes: int = Field(default=4096, gt=0)
    # Also covers consolidating user.md: when false, shared facts are written unseen.
    require_shared_approval: bool = True
    # Share of each prompt's budget the memory may take (private notes first).
    context_share: float = Field(default=0.2, gt=0.0, le=0.8)
    # "council": a persona only recalls notes written in the same council, so one
    # domain's notes don't bleed into another's debates. Shared facts stay global.
    scope: Literal["council", "global"] = "council"

    def notes_council(self, council: str) -> str | None:
        """Return the council private notes are limited to, or None for all of them."""
        return council if self.scope == "council" else None


class DossierConfig(_Section):
    """Files the user attaches (`/attach`, `council new --attach`)."""

    # Larger files are summarized once by the moderator; smaller ones go in verbatim.
    summarize_over_tokens: int = Field(default=4000, gt=0)
    max_file_bytes: int = Field(default=1_000_000, gt=0)
    csv_head_rows: int = Field(default=10, ge=0)  # kept verbatim beside a CSV's summary
    # Share of each prompt's budget the dossier may take (small num_ctx models).
    context_share: float = Field(default=0.4, gt=0.0, le=0.8)


class DiscordConfig(_Section):
    """`council chat discord` (M16 part B); lives in config.local.toml, never config.toml."""

    token: SecretStr  # never logged or shown (SecretStr hides it from repr and errors)
    guild: int  # server ID
    channel: int  # the private channel the bot reads and posts in
    owner: int  # user ID: starts questions, /run, /close, /offline
    allowed_users: list[int] = Field(default_factory=list)  # friends: speak and answer
    # Optional nicknames by user ID ("234…" = "Ben"); default: the Discord username.
    names: dict[str, str] = Field(default_factory=dict)
    council: str = "finance"  # used by /ask when no council is given
    question_timeout_s: float = Field(default=600, gt=0)  # then the debate goes on
    max_lines_per_round: int = Field(default=10, ge=1)  # per person, between updates
    max_message_chars: int = Field(default=2000, ge=1)  # a person's line, cut past this


class ChatConfig(_Section):
    """Chat front ends; absent sections mean the front end is not set up."""

    discord: DiscordConfig | None = None


class Settings(_Section):
    """Full application settings."""

    root: Path = Field(default=Path("."), exclude=True)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    dossier: DossierConfig = Field(default_factory=DossierConfig)
    user: UserConfig = Field(default_factory=UserConfig)
    local_only: LocalOnlyConfig = Field(default_factory=LocalOnlyConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path against the config root."""
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def personas_dir(self) -> Path:
        """Return the absolute personas directory."""
        return self.resolve(self.paths.personas)

    @property
    def councils_dir(self) -> Path:
        """Return the absolute councils directory."""
        return self.resolve(self.paths.councils)

    @property
    def user_name(self) -> str:
        """Return the user's name: [user] name, else the capitalized login name."""
        if self.user.name:
            return self.user.name
        try:
            login = getpass.getuser()
        except (OSError, KeyError):  # no login name in the environment or passwd
            login = ""
        return login[:1].upper() + login[1:] if login else DEFAULT_USER_NAME

    @property
    def data_dir(self) -> Path:
        """Return the absolute data directory."""
        return self.resolve(self.paths.data)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return base with override merged in recursively (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file, or return {} if it doesn't exist."""
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc


def load_settings(root: Path | None = None) -> Settings:
    """Load settings from `root` (default: cwd); missing files mean defaults."""
    root = (root or Path.cwd()).resolve()
    data = deep_merge(_read_toml(root / CONFIG_FILE), _read_toml(root / LOCAL_CONFIG_FILE))
    try:
        return Settings.model_validate({**data, "root": root})
    except ValidationError as exc:
        raise ConfigError(f"{root / CONFIG_FILE} (+ local override): {exc}") from exc
