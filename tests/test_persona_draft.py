"""Persona generator (M13): drafting, assembling, validating and writing a persona file."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from privy_council import cli
from privy_council.backends.base import BackendRateLimitError
from privy_council.backends.fake import FakeBackend, Reply
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings
from privy_council.engine.budget import Ledger
from privy_council.loaders.errors import LoadError
from privy_council.loaders.personas import load_persona
from privy_council.models.persona import BackendKind, Persona
from privy_council.persona_draft import (
    DEFAULT_SEAT_MODEL,
    DraftError,
    PersonaDraft,
    Seat,
    check_persona,
    draft_persona,
    draft_request,
    render_persona,
    write_persona,
)
from tests.conftest import REPO_ROOT
from tests.engine.test_protocol import personas as personas

DESCRIPTION = "a frugal retired engineer who hates subscriptions"


def draft(**changes: Any) -> dict[str, Any]:
    """A valid structured draft, as the model would return it."""
    return {
        "name": "Thrift",
        "slug": "thrift",
        "role": "advisor",
        "color": "#98c379",
        "expertise": ["household budgets", "repairs"],
        "tags": ["finance", "purchases"],
        "who": "You are Thrift, a retired engineer.",
        "think": ["You count the total cost.", "- You never make up a price."],
        "biases": ["You undervalue convenience. Say so when relevant."],
        "speak": "Plain and brief. Concise.",
        **changes,
    }


def test_request_carries_the_description_in_the_prompt(personas: dict[str, Persona]) -> None:
    req = draft_request(DESCRIPTION, personas, ["finance", "purchases"], "sonnet")
    [message] = req.messages
    assert DESCRIPTION in message.content and DESCRIPTION not in req.system
    assert "You are Rigor" in message.content  # the example file
    assert "rigor" in message.content and "#e5c07b" in message.content  # taken slugs, colors
    assert req.tag.purpose == "persona_draft" and req.tag.session is None
    assert req.model == "sonnet" and not req.tools
    assert req.json_schema is not None
    assert set(req.json_schema["required"]) == set(req.json_schema["properties"])


async def test_draft_is_parsed_or_kept_raw(personas: dict[str, Persona]) -> None:
    req = draft_request(DESCRIPTION, personas, [], "sonnet")
    backend = FakeBackend([draft(), draft(role="moderator")])
    assert (await draft_persona(backend, req)).name == "Thrift"
    with pytest.raises(DraftError) as caught:
        await draft_persona(backend, req)
    assert '"role": "moderator"' in caught.value.raw


def test_render_check_and_write(tmp_path: Path, personas: dict[str, Persona]) -> None:
    text = render_persona(PersonaDraft.model_validate(draft()), Seat(), ["finance"])
    assert text.startswith("---\nname: Thrift\nslug: thrift\ncolor: '#98c379'\n")
    assert "expertise: [household budgets, repairs]" in text
    assert "- You never make up a price.\n" in text  # no doubled bullet
    persona = check_persona(text, "thrift", tmp_path, personas)
    assert persona.meta.backend is BackendKind.OLLAMA
    assert persona.meta.model == DEFAULT_SEAT_MODEL and persona.meta.tags == ["finance"]
    assert persona.body.startswith("# Who you are\nYou are Thrift")

    write_persona(text, tmp_path / "thrift.md")
    assert load_persona(tmp_path / "thrift.md") == persona
    with pytest.raises(FileExistsError):
        write_persona("other", tmp_path / "thrift.md")
    with pytest.raises(ValueError, match="already exists"):
        check_persona(text, "thrift", tmp_path, personas)


def test_check_refuses_taken_names_and_bad_fields(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    def text(**changes: Any) -> str:
        """Render a draft with changes."""
        return render_persona(PersonaDraft.model_validate(draft(**changes)), Seat())

    with pytest.raises(ValueError, match="slug 'rigor' already exists"):
        check_persona(text(slug="rigor"), "rigor", tmp_path, personas)
    with pytest.raises(ValueError, match="nick 'Rigor' is already used"):
        check_persona(text(name="Rigor"), "thrift", tmp_path, personas)
    with pytest.raises(LoadError, match="color"):
        check_persona(text(color="green"), "thrift", tmp_path, personas)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A root with a copy of the repo's personas (the command writes there)."""
    shutil.copytree(REPO_ROOT / "personas", tmp_path / "personas")
    (tmp_path / "config.toml").write_text(
        f"[paths]\ncouncils = '{REPO_ROOT / 'councils'}'\ndata = '{tmp_path / 'data'}'\n"
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path


async def test_a_slug_that_is_a_path_is_refused(personas: dict[str, Persona]) -> None:
    req = draft_request(DESCRIPTION, personas, [], "sonnet")
    backend = FakeBackend([draft(slug="../../../../.claude/CLAUDE")])
    with pytest.raises(DraftError) as caught:
        await draft_persona(backend, req)
    assert "../../../../.claude/CLAUDE" in caught.value.raw
    assert "pattern" not in str(req.json_schema)  # the schema sent to claude is unchanged


def with_claude(monkeypatch: pytest.MonkeyPatch, replies: list[Reply]) -> FakeBackend:
    """Route the CLI's Claude calls to a scripted fake."""
    fake = FakeBackend(replies, default=None)

    def registry(settings: Settings, ledger: Ledger, local_only: bool = False) -> BackendRegistry:
        """A registry whose Claude backend is the fake."""
        return BackendRegistry(
            settings, ledger, overrides={BackendKind.CLAUDE_CLI: fake}, local_only=local_only
        )

    monkeypatch.setattr(cli, "BackendRegistry", registry)
    return fake


def council(root: Path, *args: str, input: str | None = None) -> Result:
    """Run `council --root <root> ...`."""
    return CliRunner().invoke(cli.app, ["--root", str(root), *args], input=input)


def test_cli_writes_after_confirmation(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = with_claude(monkeypatch, [draft(), draft()])
    declined = council(root, "personas", "new", DESCRIPTION, input="n\n")
    assert declined.exit_code == 0, declined.output
    assert not (root / "personas" / "thrift.md").exists()
    assert (root / "data" / "exports" / "persona-draft-thrift.md").is_file()

    done = council(
        root, "personas", "new", DESCRIPTION, "--slug", "miser", "--model", "gemma4:26b", "--yes"
    )
    assert done.exit_code == 0, done.output
    persona = load_persona(root / "personas" / "miser.md")
    assert persona.name == "Thrift" and persona.meta.model == "gemma4:26b"
    assert "add miser to a council's members" in done.output
    assert len(fake.requests) == 2


def test_cli_refusals(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = with_claude(
        monkeypatch, [draft(name="Rigor"), BackendRateLimitError("usage limit reached")]
    )
    taken = council(root, "personas", "new", DESCRIPTION, "--slug", "rigor")
    assert taken.exit_code == 1 and "already exists" in taken.output
    assert fake.requests == []  # refused before spending the call

    nick = council(root, "personas", "new", DESCRIPTION, "--yes")
    assert nick.exit_code == 1 and "nick 'Rigor' is already used" in nick.output
    assert "draft kept in" in nick.output

    limited = council(root, "personas", "new", DESCRIPTION)
    assert limited.exit_code == 1 and "Claude call failed: usage limit reached" in limited.output


def test_drafts_with_a_path_slug_stay_in_exports(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exports = root / "data" / "exports"
    with_claude(monkeypatch, [draft(slug="../../escaped")])
    by_model = council(root, "personas", "new", DESCRIPTION, "--yes")
    assert by_model.exit_code == 1 and (exports / "persona-draft-unnamed.md").is_file()
    (exports / "persona-draft-unnamed.md").unlink()

    with_claude(monkeypatch, [draft()])
    by_owner = council(root, "personas", "new", DESCRIPTION, "--slug", "../../escaped", "--yes")
    assert by_owner.exit_code == 1 and (exports / "persona-draft-unnamed.md").is_file()
    assert not (root / "data" / "escaped.md").exists() and not (root / "escaped.md").exists()
    assert sorted(p.name for p in exports.iterdir()) == ["persona-draft-unnamed.md"]
