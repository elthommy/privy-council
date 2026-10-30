"""Shared state and plumbing of a `DebateEngine`: events, persistence, backend calls.

`EngineCore` holds what every part of the engine uses. The protocol
(`protocol.py`), research (`research_flow.py`) and memory (`memory_flow.py`)
build on it; nothing here calls into them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from privy_council.backends.base import (
    BackendRateLimitError,
    CallTag,
    ChatMessage,
    CompletionRequest,
    CompletionResult,
    TextSink,
)
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings
from privy_council.engine import context, prompts
from privy_council.engine.budget import BudgetReport, cap_warnings, quota_warnings
from privy_council.engine.notices import (
    NOTICE_CUT_OFF,
    NOTICE_FALLBACK_ON,
    NOTICE_LANGUAGE_CURRENT,
    NOTICE_LANGUAGE_LOCKED,
    NOTICE_LANGUAGE_SET,
    NOTICE_LANGUAGE_USAGE,
    NOTICE_NEEDS_TOPIC,
    NOTICE_TOPIC_SET,
    ROUND_LABEL_BLIND,
    ROUND_LABEL_CROSS,
    ROUND_LABEL_OPEN,
)
from privy_council.engine.research import ResearchMode
from privy_council.engine.resume import messages_of, replay_events, resume_state
from privy_council.engine.state_machine import StateMachine
from privy_council.memory.store import MemoryStore
from privy_council.models.council import CouncilPreset, MemoryMode
from privy_council.models.persona import BackendKind, Persona
from privy_council.models.session import (
    ErrorEvent,
    Event,
    Message,
    MessageEvent,
    PendingTurn,
    Phase,
    SessionMeta,
    SessionState,
    SessionStateChanged,
    TextDelta,
    TurnEnded,
)
from privy_council.search import SearchProvider, SearxngProvider
from privy_council.storage.sessions import SessionStore
from privy_council.util.language import (
    LANGUAGE_NAMES,
    LANGUAGES,
    Language,
    detect_language,
    parse_language,
)
from privy_council.util.tokens import estimate_tokens

log = logging.getLogger(__name__)

# Calls that keep the language they're given: a consolidation rewrites notes from
# sessions in either language and must not translate them.
_LANGUAGE_FREE_PURPOSES = frozenset({"consolidate"})
# Prompt tokens kept free for the language line (added after the prompt is built).
_LANGUAGE_RESERVE_TOKENS = 2 * max(
    estimate_tokens(prompts.language_line(lang, structured=True)) for lang in LANGUAGES
)

Listener = Callable[[Event], None]


class EngineCore:
    """Session state, events and backend calls shared by the engine parts."""

    def __init__(
        self,
        *,
        settings: Settings,
        council: CouncilPreset,
        personas: Mapping[str, Persona],
        registry: BackendRegistry,
        topic: str,
        store: SessionStore | None = None,
        research: ResearchMode = ResearchMode.ON_DEMAND,
        memory: MemoryStore | None = None,
        restored: tuple[SessionMeta, list[Event]] | None = None,
        lang: Language | None = None,
        search: SearchProvider | None = None,
    ) -> None:
        """Set up a new session (state CREATED); nothing runs until `step()`.

        `lang` fixes the debate language; None detects it from the question.
        `search` is the local web search for local-only sessions; None uses SearXNG
        when `[research.local]` is enabled.
        `restored` (saved meta and transcript events) rebuilds a saved session
        instead; prefer `DebateEngine.resume`.
        """
        self.settings = settings
        self.council = council
        self.registry = registry
        self.store = store
        self.memory = memory or MemoryStore(settings.data_dir)
        local = settings.research.local
        self.search = search or (SearxngProvider(local) if local.enabled else None)
        self.user_name = settings.user_name
        if store:
            store.nicks["user"] = self.user_name
        if restored is None:
            self.meta = SessionMeta(
                id=store.id if store else "unsaved",
                council=council.name,
                topic=topic,
                moderator=council.moderator,
                members=list(council.members),
                prompts_version=prompts.PROMPTS_VERSION,
                research=research.value,
                lang=lang or (detect_language(topic) if topic.strip() else None),
                local_only=registry.local_only,
            )
            events: list[Event] = []
        else:
            self.meta, events = restored
            self.meta.state = resume_state(self.meta.state)
        self.research_mode = ResearchMode(self.meta.research)
        # From the session, not the preset: the preset may have changed since.
        self.personas = dict(personas)  # everyone loaded: `/summon` candidates, old nicks
        self.moderator = personas[self.meta.moderator]
        self.members = {slug: personas[slug] for slug in self.meta.members}
        for persona in [self.moderator, *self.members.values()]:
            self.meta.record_model(persona)
        self.machine = StateMachine(self.meta.state)
        self.messages: list[Message] = messages_of(events)
        self.history: list[Event] = replay_events(events)  # shown again by front ends
        self.resumed = restored is not None
        self.listeners: list[Listener] = []
        self.claude_exhausted = False
        self.busy = False
        self._pause_requested = False
        self._forget_pending: str | None = None
        self._asked_user = False  # set when the moderator forwards questions to the user
        # Each member turn's (sources, given) by (slug, round), for the moderator's flags.
        self._turn_sources: dict[tuple[str, int], tuple[str, str]] = {}
        self._save()

    @property
    def state(self) -> SessionState:
        """Return the current state machine state."""
        return self.machine.state

    @property
    def phase(self) -> Phase:
        """Return what the next `step()` will do."""
        return self.meta.runtime.phase

    @property
    def live_log(self) -> Path | None:
        """Return the live Markdown transcript path, if the session is saved."""
        return self.store.transcript_md if self.store else None

    def subscribe(self, listener: Listener) -> None:
        """Register a listener called synchronously for every event."""
        self.listeners.append(listener)

    def nick(self, slug: str) -> str:
        """Return the display nick for a speaker slug."""
        if slug == "user":
            return self.user_name
        persona = self.persona_of(slug)
        return persona.name if persona else slug

    @property
    def guests(self) -> list[str]:
        """Return the people besides the owner who wrote in the chat, first seen first."""
        names = (m.author for m in self.messages if m.speaker == "user" and m.author)
        return [n for n in dict.fromkeys(names) if n != self.user_name]

    def persona_of(self, slug: str) -> Persona | None:
        """Return the persona behind a speaker slug, including members who were kicked."""
        return self.personas.get(slug)

    @property
    def awaiting_topic(self) -> bool:
        """Return whether the session still needs its question (`council new` without one)."""
        return not self.meta.topic.strip()

    def set_topic(self, text: str) -> bool:
        """Set the question of a session started without one; return whether it was set."""
        topic = text.strip()
        if not self.awaiting_topic or not topic:
            return False
        self.meta.topic = topic
        if self.meta.lang is None:
            self.meta.lang = detect_language(topic)
        if self.store:
            self.store.retopic(self.council.name, topic)
            self.meta.id = self.store.id
        self._save()
        self._notice(NOTICE_TOPIC_SET)
        return True

    def set_language(self, arg: str) -> None:
        """`/lang [en|fr]`: show the debate language, or change it until a member speaks."""
        name = LANGUAGE_NAMES[self.meta.language]
        if not arg.strip():
            self._notice(NOTICE_LANGUAGE_CURRENT.format(name=name))
            return
        lang = parse_language(arg)
        if lang is None:
            self._notice(NOTICE_LANGUAGE_USAGE)
        elif any(
            m.speaker in self.personas and m.speaker != self.moderator.slug for m in self.messages
        ):
            # Members' messages in the old language would fill every later context, and
            # small models follow what they read over what they're told.
            self._notice(NOTICE_LANGUAGE_LOCKED.format(name=name))
        else:
            self.meta.lang = lang
            self._save()
            self._notice(NOTICE_LANGUAGE_SET.format(name=LANGUAGE_NAMES[lang]))

    def usage(self) -> BudgetReport | None:
        """Return Claude usage for this session, today and the week, if a ledger is set."""
        ledger = self.registry.ledger
        return ledger.report(session=self.meta.id) if ledger else None

    def budget_warnings(self, estimated_calls: int) -> list[str]:
        """Return soft-cap and quota warnings for `estimated_calls` more Claude calls."""
        report = self.usage()
        if report is None:
            return []
        claude = self.settings.claude
        return cap_warnings(
            report, estimated_calls, claude.session_soft_cap_calls, claude.daily_soft_cap_calls
        ) + quota_warnings(report.quota, claude.quota_warn_utilization)

    def _localized(
        self, system: str, messages: list[ChatMessage], purpose: str, structured: bool = False
    ) -> tuple[str, list[ChatMessage]]:
        """Append the debate language line to the system prompt and the last user message."""
        if purpose in _LANGUAGE_FREE_PURPOSES:
            return system, messages
        line = prompts.language_line(self.meta.language, structured)
        last = len(messages) - 1 - [m.role for m in reversed(messages)].index("user")
        chat = list(messages)
        chat[last] = ChatMessage(role="user", content=f"{messages[last].content}\n\n{line}")
        return f"{system}\n\n{line}", chat

    def _max_output(self, persona: Persona) -> int:
        """Return the persona's hard output cap, or the engine default."""
        return persona.meta.max_output_tokens or self.settings.engine.max_output_tokens

    def _budget(self, persona: Persona) -> int:
        """Return the prompt token budget for the persona's current route."""
        route = self.registry.route(persona, self.claude_exhausted)
        budget = context.input_budget(
            self.settings, route.kind, route.model, self._max_output(persona)
        )
        return budget - _LANGUAGE_RESERVE_TOKENS

    def _placement(self, turn: PendingTurn) -> tuple[BackendKind, str]:
        """Return where a turn will run, for scheduling."""
        return self._route_placement(self.members[turn.speaker])

    async def _complete(
        self,
        persona: Persona,
        prompt: context.Prompt,
        purpose: str,
        *,
        messages: list[ChatMessage] | None = None,
        json_schema: dict[str, Any] | None = None,
        tools: list[str] | None = None,
        timeout_s: float | None = None,
        max_output: int | None = None,
        model: str | None = None,
        stream: bool = False,
    ) -> tuple[CompletionResult, bool]:
        """Call the persona's backend; returns the result and whether it was a fallback.

        `stream` sends the reply to listeners as `TextDelta`s while it is generated.

        On a Claude usage limit with fallback enabled, re-route to the local model,
        except for tool calls (web research), which only the claude CLI can serve.
        `model` overrides the persona's model (e.g. synthesis_model), but never the
        local fallback model.
        """
        route = self.registry.route(persona, self.claude_exhausted)
        system, chat = self._localized(
            prompt.system,
            messages or [ChatMessage(role="user", content=prompt.user)],
            purpose,
            structured=json_schema is not None,
        )
        req = CompletionRequest(
            system=system,
            messages=chat,
            model=route.model if route.fallback or model is None else model,
            temperature=persona.meta.temperature,
            max_output_tokens=max_output or self._max_output(persona),
            json_schema=json_schema,
            tools=tools or [],
            timeout_s=timeout_s,
            tag=CallTag(session=self.meta.id, persona=persona.slug, purpose=purpose),
        )
        if prompt.dropped_messages or prompt.state_clipped:
            # Visible with DEBUG logging: how often small num_ctx windows lose context.
            log.debug(
                "trimmed %s/%s (%s, round %d, %s): %d message(s) dropped, state clipped: %s",
                persona.slug,
                purpose,
                req.model,
                self.meta.runtime.round,
                route.kind.value,
                prompt.dropped_messages,
                prompt.state_clipped,
            )
        on_text = self._typing_sink(persona.slug) if stream else None
        try:
            return await route.backend.complete(req, on_text), route.fallback
        except BackendRateLimitError:
            if (
                route.kind is not BackendKind.CLAUDE_CLI
                or not self.settings.claude.fallback_to_local
                or tools
            ):
                raise
            if not self.claude_exhausted:
                self.claude_exhausted = True
                self._notice(NOTICE_FALLBACK_ON)
            return await self._complete(
                persona,
                prompt,
                purpose,
                messages=messages,
                json_schema=json_schema,
                model=model,
                stream=stream,
            )

    def _typing_sink(self, speaker: str) -> TextSink:
        """Return a sink that emits a speaker's streamed text; its first chunk restarts."""
        first = True

        def sink(text: str) -> None:
            """Emit one chunk."""
            nonlocal first
            self._emit(TextDelta(speaker=speaker, text=text, restart=first))
            first = False

        return sink

    def _facts(self) -> str | None:
        """Return the session's web research for prompts, or None."""
        return context.facts_text(self.meta.runtime.research)

    def _dossier(self) -> str | None:
        """Return the files the user attached, for prompts, or None."""
        return context.dossier_text(self.meta.runtime.dossier)

    def _memory(self, persona: Persona) -> tuple[str, str]:
        """Return (private notes, shared facts) for a persona's prompt.

        Read on every call: the files are tiny, and `/forget` takes effect at once.
        """
        if self.council.memory is MemoryMode.OFF:
            return "", ""
        if not persona.meta.memory:
            return "", self.memory.read_shared()
        council = self.settings.memory.notes_council(self.council.name)
        return self.memory.read_private(persona.slug, council), self.memory.read_shared()

    def _needs_topic(self) -> bool:
        """Say so and return True when the session has no question yet."""
        if self.awaiting_topic:
            self._notice(NOTICE_NEEDS_TOPIC)
        return self.awaiting_topic

    def _speakers(self) -> list[Persona]:
        """Return the moderator and the members."""
        return [self.moderator, *self.members.values()]

    def _route_placement(self, persona: Persona) -> tuple[BackendKind, str]:
        """Return where a persona's calls run now, for scheduling."""
        route = self.registry.route(persona, self.claude_exhausted)
        return route.kind, route.model

    def _member_system(self, persona: Persona) -> str:
        """Return a member's system prompt: persona body plus engine rules."""
        rules = prompts.member_protocol(
            persona, self.moderator, self.user_name, self.meta.language, self.guests
        )
        return f"{persona.body}\n\n{rules}"

    def _moderator_system(self) -> str:
        """Return the moderator's system prompt: persona body, council brief, engine rules."""
        members = list(self.members.values())
        rules = prompts.moderator_protocol(members, self.user_name, self.guests)
        brief = self.council.moderator_brief.strip()
        body = f"{self.moderator.body}\n\n{brief}" if brief else self.moderator.body
        return f"{body}\n\n{rules}"

    def _is_blind(self) -> bool:
        """Return whether the current round is the blind first round."""
        return self.meta.runtime.round == 1 and self.council.protocol.blind_first_round

    def _round_label(self) -> str:
        """Return the chat label for the round that is starting."""
        n = self.meta.runtime.round
        if self._is_blind():
            return ROUND_LABEL_BLIND
        return ROUND_LABEL_OPEN.format(round=n) if n == 1 else ROUND_LABEL_CROSS.format(round=n)

    def _post_result(self, persona: Persona, result: CompletionResult, fallback: bool) -> None:
        """Post a model reply, flagging fallback and truncation."""
        self._post(persona.slug, result.text.strip(), fallback=fallback, truncated=result.truncated)
        if result.truncated:
            self._notice(NOTICE_CUT_OFF.format(nick=persona.name))

    def _post(
        self,
        speaker: str,
        text: str,
        *,
        fallback: bool = False,
        truncated: bool = False,
        to_user: bool = False,
        author: str | None = None,
    ) -> None:
        """Append a message to the transcript and emit it."""
        message = Message(
            speaker=speaker,
            text=text,
            round=self.meta.runtime.round,
            fallback=fallback,
            truncated=truncated,
            to_user=to_user,
            author=author,
        )
        self.messages.append(message)
        self._emit(MessageEvent(message=message))

    def _notice(self, text: str) -> None:
        """Post a system line."""
        self._post("system", text)

    def _fail(self, text: str) -> None:
        """Report a recoverable failure and park the session in ERROR_RECOVERABLE."""
        self._emit(ErrorEvent(text=text))
        self._set_state(SessionState.ERROR_RECOVERABLE)

    def _set_state(self, target: SessionState) -> None:
        """Transition the state machine, persist and announce it."""
        self.machine.to(target)
        self.meta.state = target
        self._save()
        self._emit(SessionStateChanged(state=target))

    def _emit(self, event: Event) -> None:
        """Persist the event (UI-only signals excepted), then hand it to listeners."""
        if self.store and not isinstance(event, TextDelta | TurnEnded):
            self.store.append(event)
        for listener in self.listeners:
            listener(event)

    def _save(self) -> None:
        """Persist session.json."""
        if self.store:
            self.store.save_meta(self.meta)
