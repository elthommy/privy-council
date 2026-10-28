"""All engine prompt text. Persona prompts live in personas/*.md, never here."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from privy_council.backends.base import ChatMessage
from privy_council.engine.dynamics import REPEAT_THRESHOLD, MemberStats
from privy_council.models.flaw import Flaw, FlawKind
from privy_council.models.persona import Persona, Role
from privy_council.models.session import DebateState, Message
from privy_council.models.vote import Vote
from privy_council.util.language import Language

PROMPTS_VERSION = "48"

# Backends that accept a single user turn (the claude CLI on stdin) flatten a
# multi-message conversation with these labels.
ROLE_LABELS = {"user": "User", "assistant": "Assistant"}
FLATTENED_MESSAGE = "[{label}]\n{content}"


def flatten_messages(messages: Sequence[ChatMessage]) -> str:
    """Render messages as one text block; a lone user message is passed through as is."""
    if len(messages) == 1 and messages[0].role == "user":
        return messages[0].content
    return "\n\n".join(
        FLATTENED_MESSAGE.format(label=ROLE_LABELS[m.role], content=m.content) for m in messages
    )


# --- Debate language (M10) -------------------------------------------------------

# Appended by the engine to the system prompt and to the end of the user message of
# every call (consolidation excepted). Small models drift back to the language of
# the instructions, so French sessions say it plainly, last.
LANGUAGE_LINES: dict[Language, str] = {
    "en": "Write your reply in English.",
    "fr": (
        "Write your reply in French (en français), whatever the language of these "
        "instructions, of your persona or of earlier messages. Address one person (the user "
        "or a member) as « tu », never « vous »; « vous » is only for several people. "
        "@nicks stay as they are."
    ),
}
# Only on calls that return JSON: on a member turn, the word alone made gpt-oss
# answer in JSON.
STRUCTURED_LANGUAGE_NOTES: dict[Language, str] = {
    "en": "",
    "fr": " JSON keys and slugs stay as they are.",
}

# The fixed labels members and the moderator write, per language: a small model
# mixes English labels into French text unless the template shows French ones.
LABELS: dict[Language, dict[str, str]] = {
    "en": {
        "position": "Position:",
        "reasons": "Reasons:",
        "confidence": "Confidence:",
        "assumption": "Key assumption:",
        "update": "I update:",
        "devil": "Playing devil's advocate:",
        "the_user": "the user",
        "fact_start": "The user ...",
        "h_consensus": "Consensus",
        "h_disagreements": "Remaining disagreements",
        "h_errors": "Errors caught",
        "h_spread": "Confidence spread",
        "h_questions": "Questions for the user",
        "h_next": "Suggested next steps",
        "h_consensus_so_far": "Consensus so far",
        "h_open_disagreements": "Open disagreements",
        "h_open_questions": "Open questions",
        "research_opening": "background for the question",
    },
    "fr": {
        "position": "Position :",
        "reasons": "Raisons :",
        "confidence": "Confiance :",
        "assumption": "Hypothèse clé :",
        "update": "Je change d'avis :",
        "devil": "Je joue l'avocat du diable :",
        "the_user": "l'utilisateur",
        "fact_start": "L'utilisateur ...",
        "h_consensus": "Consensus",
        "h_disagreements": "Désaccords restants",
        "h_errors": "Erreurs relevées",
        "h_spread": "Éventail des confiances",
        "h_questions": "Questions pour l'utilisateur",
        "h_next": "Prochaines étapes suggérées",
        "h_consensus_so_far": "Consensus à ce stade",
        "h_open_disagreements": "Désaccords ouverts",
        "h_open_questions": "Questions ouvertes",
        "research_opening": "contexte de la question",
    },
}


def language_line(lang: Language, structured: bool = False) -> str:
    """Return the instruction naming the language of the reply (JSON replies: keys kept)."""
    return LANGUAGE_LINES[lang] + (STRUCTURED_LANGUAGE_NOTES[lang] if structured else "")


def task(template: str, lang: Language, **fields: object) -> str:
    """Format a task template with the language's labels and the given fields."""
    return template.format(**LABELS[lang], **fields)


# --- Member system prompt (appended to the persona body) ----------------------

ROLE_MANDATES = {
    Role.ADVISOR: "advisor. Give your honest best advice from your own expertise and angle.",
    Role.DEVIL_ADVOCATE: (
        "devil's advocate. Argue against the leading position, even if you privately "
        "agree, and flag it when you do."
    ),
    Role.RISK_OFFICER: (
        "risk officer. Name the failure modes and the worst plausible case of every "
        "option on the table."
    ),
    Role.SCRIBE: "scribe. Track what has been established and state it precisely.",
    Role.MODERATOR: "moderator. Run the debate; never give your own opinion.",
}

MEMBER_PROTOCOL = """\
# Council protocol
You are {name}, one member of a council debating the user's question in a group chat. \
A neutral moderator, {moderator}, runs the debate. The user's name is {user}; address \
them by name.
Your role: {mandate}

Rules:
- Speak only as {name}. Never write lines for other members, the moderator or the user.
- Be concise: a chat turn, not an essay.
- Don't agree just to be polite. When you agree, give a reason that is new or that you \
reinforce.
- Only if your position changed since your last message, say so: start with "{update}" \
followed by your new position. If it didn't change, don't use those words.
- Don't invent facts about the user. When the answer depends on something only the user \
knows, say what you would need to know.
- Figures: a number from the fresh facts, the dossier or the user's messages is \
sourced. Quote it as written and name its source in a few words ("$233 at Pangoly"), \
even when the source is undated or the sources disagree. "(estimate, from memory)" is \
only for a number found nowhere there; then prefer a range. If another member labelled \
a sourced figure, quote it without the label. If you can't give a number honestly, \
argue without one.
- Don't claim what sources say, or that none say something, beyond the facts in front \
of you."""

# --- Moderator system prompt (appended to the moderator persona body) ---------

MODERATOR_PROTOCOL = """\
# Your job as moderator
You run a debate between these council members (slug, @nick, role, expertise):
{roster}

You never give your own opinion on the question. Address members by @nick. The user's \
name is {user}; address them by name."""

ROSTER_LINE = "- {slug} (@{name}): {role}{expertise}"

# Appended to both protocols only when someone besides the owner has written in the chat,
# so a one-person session keeps the prompts above word for word.
PARTICIPANTS_MEMBER = """\
Several people share this chat: {people}. {owner} asked the question. Address the \
person you answer by @name. A fact someone gives is about their own situation, not \
anyone else's."""

PARTICIPANTS_MODERATOR = """\
Several people share this chat: {people}. {owner} asked the question. Wherever these \
rules say "the user", they mean each of these people: user_facts holds every person's \
facts, and questions_for_user asks whichever person can answer, {guest} as much as \
{owner}. Address people by @name. In debate_state.user_facts, start each fact with the \
name of the person who \
stated it ("{owner}: ..."). Each of questions_for_user and debate_state.open_questions \
is one question that starts with the @name of the one person who can answer it. A \
question about someone's own money, income, plans or wishes goes to that person \
("@{guest} how much can you put in?"), never to {owner} on their behalf; ask @{owner} \
only about {owner}'s own situation or a choice the whole group makes. Never put two \
@names in one question: ask each person separately. Once a person has answered a \
question in the chat, drop it from open_questions and don't ask it again.
Read every line the people wrote since your last update: each new fact goes into \
user_facts under that person's name, and a person who changes their mind corrects \
their earlier fact (replace it, don't keep both). A person's opinion is their own fact \
("{guest}: wants ..."), never a member's stance: disagreements.sides holds member \
slugs only, each with what that member argued. Never credit one person's or member's \
view to another."""

# How many questions the update may ask: in total for one person, per person in a shared chat.
QUESTIONS_CAP_ONE = "2 questions"
QUESTIONS_CAP_SHARED = "2 questions per person, each to one person by @name"

# Appended to the round update's task in a shared chat when people wrote since the last
# update: buried among the members' turns, a local Chair missed these (M16 full check).
PEOPLE_LINES_TASK = """\
The people wrote these lines since your last update. Go through them first: add each \
new fact to user_facts under the person's name, replace a fact the person changed or \
corrected, and drop from open_questions every question they answered.
{lines}"""

# --- Per-turn user prompt sections --------------------------------------------

SECTION = "# {title}\n{body}"
TITLE_QUESTION = "Question"
TITLE_STATE = "Debate state (maintained by the moderator)"
TITLE_HISTORY = "Recent messages"
TITLE_ROUND_MESSAGES = "Messages since your last update"
TITLE_TASK = "Your task"
CHAT_LINE = "<{nick}> {text}"
USER_NICK = "User"
TRIMMED_MARK = "\n[…trimmed]"
EMPTY = "(nothing yet)"

STATE_USER_FACTS = "What the user said about their situation:"
STATE_POSITIONS = "Positions:"
STATE_AGREEMENTS = "Agreements:"
STATE_DISAGREEMENTS = "Disagreements:"
STATE_OPEN_QUESTIONS = "Open questions for the user:"
STATE_BULLET = "- {text}"
STATE_SIDE = "{nick}: {stance}"

# --- Tasks ---------------------------------------------------------------------

OPENING_TASK = """\
Open the debate on the question above, in a few lines for the chat:
1. Restate the question in one sentence.
2. List the 2-4 key sub-questions.
3. Say what a good answer must cover.
Don't answer the question yourself, and don't call on anyone: every member answers \
the first round independently."""

BLIND_TASK = """\
Round 1, blind: give your independent take. You have not seen the other members' \
answers; don't guess them. Use this shape:
**{position}** one or two sentences.
**{reasons}** your top 3, as a numbered list.
**{confidence}** 0-100.
**{assumption}** the one assumption that, if wrong, would most change your answer."""

# Round 1 for a devil's advocate: it speaks last and sees the blind takes, so it knows
# which position leads (blind, it guessed the majority wrong).
BLIND_DEVIL_ADVOCATE_TASK = """\
Round 1: the other members answered independently; their takes are above. Play your \
role: argue against the leading position (the view most of them share), even if you \
agree with it. Use this shape, with "{devil}" written out as shown:
**{position}** {devil} one or two sentences.
**{reasons}** your top 3, as a numbered list.
**{confidence}** 0-100.
**{assumption}** the one assumption that, if wrong, would most change your answer."""

OPEN_ROUND_TASK = """\
Round {round}: give your take, reacting to what others said if relevant. Use this shape:
**{position}**, **{reasons}** (top 3), **{confidence}** (0-100), **{assumption}**."""

CROSS_TASK = """\
Round {round}. @{moderator} asks you: {prompt}
Reply in a few sentences. Address members by name, e.g. @{example}."""

ASK_TASK = """\
The user asks you directly, outside the rounds: {question}
Answer in a few sentences, as yourself."""

# Floor prompts the engine writes itself, so per language: they show in the chat
# the members read, and an English line amid French ones invites drift.
FALLBACK_TURN_PROMPT: dict[Language, str] = {
    "en": "Respond to the strongest argument against your current position, or update it.",
    "fr": ("Réponds à l'argument le plus fort contre ta position actuelle, ou fais-la évoluer."),
}
# The floor prompt of a member the moderator left out for too long (M7 guard).
SILENT_TURN_PROMPT: dict[Language, str] = {
    "en": (
        "you haven't spoken for a few rounds. Respond to the point in the debate state you "
        "disagree with most, or say what the others are missing."
    ),
    "fr": (
        "tu n'as pas parlé depuis quelques tours. Réponds au point de l'état du débat avec "
        "lequel tu es le moins d'accord, ou dis ce qui manque aux autres."
    ),
}

# What the moderator says in the chat when handing a member the floor. A colon,
# not a comma: the moderator often starts its prompt with a capital.
FLOOR_LINE = "@{nick}: {prompt}"
# The floor prompt of a persona summoned after the blind round (`/summon`).
JOIN_PROMPT: dict[Language, str] = {
    "en": (
        "you just joined the channel. From the debate so far, give your own take on the "
        "question and say where you agree or disagree with the others."
    ),
    "fr": (
        "tu viens de rejoindre le salon. D'après le débat jusqu'ici, donne ton propre avis "
        "sur la question et dis où tu es d'accord ou non avec les autres."
    ),
}
# The floor prompt asking a member to check an error the moderator caught (M11), and
# the planned prompt it carries when that member was due to speak anyway.
FLAW_TURN_PROMPT: dict[Language, str] = {
    "en": (
        'you wrote "{quote}". {note} Check it and correct it if it is wrong; say plainly '
        "whether it stands."
    ),
    "fr": (
        "tu as écrit « {quote} ». {note} Vérifie-le et corrige-le s'il est faux ; dis "
        "clairement s'il tient."
    ),
}
FLAW_TURN_THEN: dict[Language, str] = {"en": " Then: {prompt}", "fr": " Ensuite : {prompt}"}
# The moderator forwarding questions to the user (one line per question).
USER_QUESTION_LINE = "@{nick} {question}"

MODERATE_TASK = """\
{dynamics}

Round {round} of at most {max_rounds} just ended. Update the debate state and plan the \
next round. Return a JSON object matching the schema. Keep it compact: every position, \
stance, agreement and question is one short sentence.
- debate_state.user_facts: every fact the user stated about their own situation, in the \
question or their messages, one short line each (e.g. "shaves every 4 days", "fixed \
costs: 1,900 € a month"). Copy the earlier user_facts word for word and add the new \
ones; change or drop one only when the user corrected it. Never a member's guess.
- debate_state.positions: each member's current position in one sentence, keyed by slug.
- debate_state.agreements: points the members actually agree on.
- debate_state.disagreements: the 1-3 sharpest disagreements; `sides` maps slug to stance.
- debate_state.open_questions: what only the user can answer, plus your own inferences.
- consensus: "full" if every member now holds essentially the same position; "partial" \
if most agree on the main answer but differ on important details, or one dissents; \
else "none". Judge from what members argued, not from polite agreement.
- new_arguments: what this round added that the debate state didn't already hold, one \
short line each naming who, e.g. "@Rigor: a layoff would also cut the card's credit \
line". Not new: restating a point, putting a number on a point already argued, \
narrowing a range, or giving a known point more weight. An empty list if nothing new.
- position_changes: slugs of the members whose position really changed this round: a \
new answer, a moved number or range, or a conceded point. Not a change: restating a \
position, "no change", or only saying what would change their mind. The dynamics show \
who wrote the update marker; that marker is often wrong, so judge from what they said. \
An empty list if nobody changed.
- decision_open: true if a remaining disagreement would change what the user should \
actually do; false if the members now differ only on details (a slightly different \
amount or timing) that leave the recommended course of action the same.
- next_turns: who speaks next and on what, at most one turn per member; never empty, \
even when you stop. `speaker` is a slug; `prompt` is your instruction, e.g. \
"respond to @Compounder's claim that ...". \
Target the sharpest disagreements and make members who disagree answer each other. \
Use the dynamics above: give the floor to members who have been quiet, and call on a \
member who keeps repeating themselves only to confront them with something new.
- stop / stop_reason: {stop_rule}
- questions_for_user: at most {questions_cap} where the user's answer would change the \
advice, rephrased for the user from what members said they need to know or from \
open_questions. Only what the user personally knows (their situation, goals, \
preferences), never a fact the web can answer. Leave it empty if nothing is that \
important. {asked_rule}
- research: {research_rule}
- summon: {summon_rule}
- flaws: clear errors in this round's member messages, most serious first, at most 4. \
`member` is a slug; `kind` is "arithmetic" (a wrong sum, conversion or unit), \
"contradiction" (the member contradicts their own earlier statement), "logic" (a \
conclusion that doesn't follow, or a rule turned upside down), "unsourced" (a figure \
stated as fact that neither the facts, the dossier nor the user gave; see the engine's \
list above, if any) or "other" (another factual error); `quote` is the member's exact \
words, short; `note` says what is wrong in one sentence. Only errors you can show, \
never a difference of judgment or a labelled estimate that is plausible. Do the check \
before you report: if the quote turns out right, or right "in context", it is no flaw, \
leave it out; and never "correct" a member toward a figure nobody sourced. {flaw_rule} \
An empty list if none.

Attribution: only attribute a position or argument to a member who actually stated it \
in the messages. Put your own inferences in open_questions, never in a member's mouth.
If the user interjected, make the next round react to it."""

RESEARCH_RULE_ON = (
    "at most 2 web searches to run before the next round, for facts the debate needs "
    "that the web can answer (prices, market closes, dates, recent events, official "
    "figures). Each is a short search focus. Skip facts the research above already "
    "covers; empty if none."
)
RESEARCH_RULE_LAST = (
    "this is the last update and the debate ends after it: always empty, since no member "
    "could discuss new facts."
)
RESEARCH_RULE_OFF = (
    "web research is off: always empty. If a fact is essential, ask the user for it in "
    "questions_for_user."
)

# The moderator relaying a search it couldn't run on its own (a chat line to the user).
LOOKUP_ASK = "I'd like to look up: {focus} ({reason}). Type /research {focus} to allow it."
LOOKUP_ANSWER = "I can't look up {focus} ({reason}); answer if you know."
REASON_AUTO_LIMIT = "web searches so far this session: {count}"
REASON_BUDGET = "the Claude budget is getting tight"
REASON_OFF = "web research is off"

ASKED_RULE_NONE = "Never ask something the user already answered."
ASKED_RULE = (
    "You already asked the user: {asked}. Never ask any of these again, even reworded "
    "or combined with something new, nor anything the user already answered. If the "
    "user declined, said they have nothing to add or told you to decide with what you "
    "know, ask nothing more: state your assumption in open_questions instead."
)

STOP_RULE_AUTO = (
    "set stop to true, with a reason, only if consensus is full and nothing is left for "
    "the members to debate; otherwise false. (The engine itself ends a debate that "
    "stops bringing new arguments.)"
)
STOP_RULE_MANUAL = "always false; the user decides when to end."
STOP_RULE_FIRST_ROUND = "always false: the members haven't challenged each other yet."
STOP_RULE_LAST_ROUND = "this was the last round: set stop to true with a short reason."

FLAW_RULE_PROMPTED = (
    "The engine asks the members to check the first {cap} next round, so don't plan a "
    "turn about the same error; the others go into the final synthesis."
)
FLAW_RULE_SYNTHESIS = "They go into the final synthesis."

SUMMON_RULE_ON = (
    "null, unless the debate clearly lacks an expertise that one of these personas (not "
    "in the channel) has: {available}. Then give that persona's slug, and as the reason "
    "the expertise that is missing, in a few words. The user decides whether it joins."
)
SUMMON_RULE_OFF = "always null."
SUMMON_CANDIDATE = "{slug} (@{name}: {role}{expertise})"
# The moderator relaying its summon proposal to the user (a chat line).
SUMMON_ASK = "I'd like to bring in @{nick}: {reason}. Type /summon {slug} to allow it."

# --- Debate dynamics, counted by the engine (M7) -------------------------------

TITLE_DYNAMICS = "Debate dynamics (counted by the engine)"
DYNAMICS_TURNS = "{turns} turn(s)"
DYNAMICS_SPOKE = "spoke this round"
DYNAMICS_SILENT = "quiet for {count} round(s)"
DYNAMICS_UPDATED = "said they changed position"
DYNAMICS_REPEAT = "largely repeated an earlier message ({percent}% overlap)"
DYNAMICS_MEMBER = "- @{nick}: {facts}"
DYNAMICS_STALL = (
    "Rounds in a row before this one that made no progress (nothing new, or nothing left "
    "that changes the decision): {count}. The engine ends the debate after {limit} in a row."
)

DYNAMICS_FLAWS = (
    "Figures the engine found in none of the member's sources (the facts, the dossier, the "
    "user's and earlier messages, the debate state), or only with another unit. They may be "
    "the member's own estimates or sums: judge whether one matters for the advice."
)
DYNAMICS_FLAW = '- @{nick}: "{figure}" in "{excerpt}"{label}'
DYNAMICS_MISLABELLED = " (labelled as an estimate, though the facts give it)"
DYNAMICS_WRONG_UNIT = " (the sources give this number only with another unit)"
DYNAMICS_FLAWS_MORE = "- … and {count} more"
# Engine-flagged figures listed per update; more would crowd the moderator's prompt.
MAX_DYNAMICS_FLAWS = 8

REPAIR_TASK = """\
Your previous reply was not valid: {error}
Return only a JSON object matching the schema, with no other text."""

SYNTHESIS_TASK = """\
The debate is over ({reason}). Write the final synthesis in Markdown with these sections:
## {h_consensus}
## {h_disagreements}
(each one: who holds which side, and why)
## {h_spread}
## {h_questions}
(only what the user, or a person in the chat, can answer about their own \
situation or preferences and that would change the answer; never a question about the \
world or the analysis (dates, prices, rules, what something is), those go in the next \
steps, and none the fresh facts already answer; write "None." if there is none)
## {h_next}

Only attribute positions members actually stated. Refer to a member by nick or as \
"they", never as "he", "his", "she" or "her". Be concise."""

# The synthesis re-asked an answered question, reworded (2026-10-28 gacha run).
SYNTHESIS_ASKED = (
    "You already asked the user during the debate: {asked}. The questions section must "
    "not repeat any of them, even reworded, nor ask anything the user already answered."
)

# The moderator's flaw reports (M11), appended to the synthesis task when there are any.
SYNTHESIS_ERRORS = """\
Errors you caught in members' messages during the debate:
{lines}
Add a section "## {h_errors}" right after "## {h_disagreements}": the errors above \
that the debate never corrected, one line each (who, what was wrong, and the right \
figure or reasoning if the debate established it). Leave out the ones a member \
corrected, and the whole section if none is left. An error is not a disagreement: \
don't list it among the disagreements."""
SYNTHESIS_ERROR_LINE = '- round {round}, {nick} ({kind}): "{quote}"{note}'

SYNTHESIS_REASON_DEFAULT = "closed by the user"

# The latest `/vote`, appended to the synthesis task (its "Confidence spread").
SYNTHESIS_VOTE = """\
The user called a vote after round {round}; each member stated their own position:
{lines}"""
SYNTHESIS_VOTE_LINE = "- {nick} ({confidence}%): {position}"

# --- Taking stock mid-debate (`/vote`, `/summary`) ---------------------------------

VOTE_TASK = """\
The user calls a vote. State your current position on the question in one sentence of \
at most 25 words, and how confident you are in it from 0 to 100. Give your own view, \
not the group's. No greeting, no hedging, no reasons: the vote is a table row. Reply \
with a JSON object only: {"position": "...", "confidence": 70}"""
# The schema's description of `position`: models read it too (claude --json-schema).
VOTE_POSITION_RULE = "Your position in one sentence of at most 25 words; no greeting."

SUMMARY_TASK = """\
The user asks for an interim summary after round {round}; the debate continues \
afterwards. Write it in Markdown with these sections:
## {h_consensus_so_far}
## {h_open_disagreements}
(each one: who holds which side, and why)
## {h_open_questions}
(what the debate or the user still needs to settle)

Only attribute positions members actually stated. Refer to a member by nick or as \
"they", never as "he", "his", "she" or "her". Be concise."""

# --- Web research (moderator only) ----------------------------------------------

RESEARCH_SYSTEM = """\
You are the research assistant of a debate moderator. You search the web and report \
facts. You never give opinions or advice."""

RESEARCH_TASK = """\
Search the web for fresh, verifiable facts that matter for the question above.
Focus: {focus}
Today is {today}.
Return at most 8 bullets, about 1,200 characters in total, with no preamble:
- each bullet: the fact, its date, and its source URL;
- prefer primary sources (statistics offices, central banks, official sites, research \
institutes, academic work) over commentary;
- say when sources disagree or data is missing.
Don't make recommendations."""

RESEARCH_DEFAULT_FOCUS = "the current facts and figures the council will need"

# --- Local web search (SearXNG, local-only sessions): queries, then a summary --------
# The engine runs the searches; the model never browses. Snippets only, no pages.

LOCAL_QUERIES_SYSTEM = """\
You are the research assistant of a debate moderator. You write web search queries. \
You never give opinions or advice."""

LOCAL_QUERIES_TASK = """\
Write 1 to {count} web search queries that would find fresh, verifiable facts for the \
question above.
Focus: {focus}
Today is {today}.
- Short keyword queries, as you would type them into a search engine: 2 to 5 words, one \
topic each. Search engines need every word to match: leave out "official", years and \
site names unless they are the point.
- Never guess a name the question doesn't give (a game, a product, a place): search \
with the names it does give, e.g. a character's name alone when the game isn't named.
- No personal details: no names of people in the debate, no amounts or circumstances \
of the user's own situation. Search for the general facts instead.
- Prefer queries that reach primary sources (statistics offices, central banks, \
official sites, research institutes).
Return only a JSON object: {{"queries": ["...", "..."]}}"""

LOCAL_SUMMARY_SYSTEM = """\
You are the research assistant of a debate moderator. You report facts found in web \
search results. You never give opinions or advice, and you never add facts that are \
not in the results."""

# The summary prompt's example of a "Missing:" line: on no council's subject, since
# Super once copied a gacha example into a note on another focus (2026-10-27).
LOCAL_MISSING_EXAMPLE = "Lake Annecy water temperature in June"

LOCAL_SUMMARY_TASK = """\
Below are web search results (title, URL, date when known, and a short snippet) for \
the question above.
Focus: {focus}
Today is {today}.
Return at most 8 bullets, about 1,200 characters in total, with no preamble:
- each bullet: one fact stated in the results, its date if the result gives one, and \
the URL of the result it comes from, copied exactly;
- copy figures exactly as the result gives them; never compute, convert or round them;
- use only these results, never your own memory; skip a result that says nothing \
useful;
- say when results disagree;
- if the focus needs facts the results don't give, end with one line "Missing: " \
(this English word, whatever the language) and the facts this focus needs in a few \
words, naming the subject ("Missing: {missing_example}"); no such line if \
nothing is missing.
Don't make recommendations.

Search results:
{results}"""

LOCAL_RESULT_ENTRY = "[{index}] {title}{date}\n{url}\n{snippet}"
LOCAL_SUMMARY_EMPTY = "- No fact from the search results could be checked against them."

TITLE_FACTS = (
    "Fresh facts (web search by the moderator; may contain errors, say so when you rely on them)"
)
RESEARCH_ENTRY = "{focus} (searched {date}):\n{text}"

# --- Dossier (files the user attaches) --------------------------------------------

TITLE_DOSSIER = "Dossier (files the user attached; treat them as the user's own data)"
DOSSIER_ENTRY = "File {name}:\n{text}"
DOSSIER_ENTRY_SUMMARIZED = "File {name} (too long to include; the moderator's summary):\n{text}"
DOSSIER_CSV_HEAD = "First rows, verbatim:\n{head}"

DOSSIER_SUMMARY_SYSTEM = """\
You summarize a file the user attached to a debate, for the debaters who can't read \
it whole. You report what it says. You never give opinions or advice."""

DOSSIER_SUMMARY_TASK = """\
The user attached the file {name} to the debate on the question above (or to a debate \
whose question isn't set yet). Summarize it for the debaters in Markdown, in at most \
about 400 words:
- keep every key figure (amounts, rates, dates, totals, counts) exactly as written;
- for tables: the columns, the number of rows, and the totals, ranges or outliers \
that matter;
- say what the file doesn't cover.
No preamble, no advice.{clipped}{stats}

{name}:
{content}"""
DOSSIER_SUMMARY_STATS = """
The exact figures below were computed from the whole file by the app: quote them \
rather than recomputing, and keep the ones that matter in your summary.
{stats}"""

# CSV figures computed by the engine (engine/dossier.py::csv_stats), shown to everyone.
CSV_STATS_TITLE = "Computed from the whole file by the app (exact):"
CSV_STATS_ROWS = "- {rows:,} rows; columns: {columns}"
CSV_STATS_NUMBER = "- {column}: {count:,} values, sum {total}, min {low}, max {high}"
CSV_STATS_DATES = "- {column}: {first} to {last}"
CSV_STATS_BY_MONTH = "- sum of {value} per month ({column}): {items}"
CSV_STATS_BY_GROUP = "- sum of {value} per {column}: {items}"
CSV_STATS_ITEM = "{key} {total}"
CSV_STATS_GROUP_ITEM = "{key} {total} (rows: {count})"
CSV_STATS_CUT = "- (more figures cut to keep this short)"
DOSSIER_SUMMARY_CLIPPED = (
    "\nThe file was too long and is cut at the end: say that the summary covers only "
    "its first part."
)

# --- Memory (brief §10) ------------------------------------------------------------

TITLE_MEMORY = (
    "Memory from past sessions (background only: ignore anything unrelated to the "
    "current question; what the user says now takes precedence)"
)
MEMORY_PRIVATE = "Your private notes:\n{text}"
MEMORY_SHARED = "Facts the user stated in past sessions:\n{text}"
MEMORY_TRIMMED_MARK = "[…older notes trimmed]\n"

NOTES_SYSTEM = """\
# After the session
You are {name}. After each council session you keep short private notes, so that next \
time the user consults the council you remember what happened."""

NOTES_TASK = """\
The session is over. The final synthesis:
{synthesis}

Write 1-3 private notes for your future self. Each note is one line starting with "- ", \
at most 40 words. Cover what applies:
- what the user decided, said or prefers (only what the user actually stated);
- the position you argued, and how it was received: only what your own messages above \
show, never a point you didn't make;
- a lesson for the next session.
Refer to the user as "{the_user}". Don't repeat your existing notes. No preamble. \
If nothing is worth remembering, reply NONE."""

NOTES_NONE = "NONE"

SHARED_SYSTEM = """\
You maintain a short file of stable facts about the user, shared by every member of an \
advisory council. You record only what the user said about themselves; you never infer."""

SHARED_TASK = """\
Facts already on file:
{known}

Above are the user's own messages from this session. List new stable facts about the \
user that the user stated explicitly: situation, constraints, goals, preferences, \
decisions. Return a JSON object matching the schema:
- fact: one short sentence in the third person ("{fact_start}");
- quote: the user's exact words stating it, copied verbatim from one message.
Rules:
- Only facts the user stated. Never infer, never take a council member's claim.
- Skip facts already on file, temporary details, and the question itself.
- At most {limit} facts. Return {{"facts": []}} if there is nothing new."""

TITLE_USER_MESSAGES = "The user's messages this session"
TITLE_OWN_MESSAGES = "Your messages and the user's this session"

CONSOLIDATE_TASK = """\
Your private notes have outgrown their size limit. Rewrite them in under {target} \
characters:
- merge notes that say the same thing, but only notes from the same council;
- drop points that are stale or superseded by a newer note;
- keep the user's decisions and preferences, and lessons that still apply.
Keep the "- YYYY-MM-DD (council): " prefix on each line, with the council unchanged \
(merged notes take the newest date), and each note in the language it is written in. \
Output only the bullet lines.

The notes:
{notes}"""

CONSOLIDATE_SHARED_TASK = """\
The shared file of facts about the user has outgrown its size limit. Rewrite it in under \
{target} characters:
- merge facts that say the same thing;
- when two facts conflict, keep the newer one;
- never add a fact or change its meaning.
Keep the "- YYYY-MM-DD (council): " prefix on each line (merged facts take the newest \
date), and each fact in the language it is written in. Output only the bullet lines.

The facts:
{notes}"""


# --- Renderers -----------------------------------------------------------------


def member_protocol(
    persona: Persona,
    moderator: Persona,
    user: str = USER_NICK,
    lang: Language = "en",
    guests: Sequence[str] = (),
) -> str:
    """Return the engine rules appended to a member's persona prompt."""
    rules = MEMBER_PROTOCOL.format(
        name=persona.name,
        moderator=moderator.name,
        mandate=ROLE_MANDATES[persona.meta.role],
        user=user,
        update=LABELS[lang]["update"],
    )
    return rules + _participants(PARTICIPANTS_MEMBER, user, guests)


def _participants(template: str, owner: str, guests: Sequence[str]) -> str:
    """Return the shared-chat rules to append, or "" when the owner is alone."""
    if not guests:
        return ""
    people = ", ".join([owner, *guests])
    return "\n" + template.format(people=people, owner=owner, guest=guests[0])


def moderator_protocol(
    members: Sequence[Persona], user: str = USER_NICK, guests: Sequence[str] = ()
) -> str:
    """Return the engine rules appended to the moderator's persona prompt."""
    lines = [
        ROSTER_LINE.format(
            slug=p.slug,
            name=p.name,
            role=p.meta.role.value,
            expertise=f"; {', '.join(p.meta.expertise)}" if p.meta.expertise else "",
        )
        for p in members
    ]
    rules = MODERATOR_PROTOCOL.format(roster="\n".join(lines), user=user)
    return rules + _participants(PARTICIPANTS_MODERATOR, user, guests)


def section(title: str, body: str) -> str:
    """Render one titled prompt section."""
    return SECTION.format(title=title, body=body)


def chat_line(message: Message, nick_of: Callable[[str], str]) -> str:
    """Render one chat message as `<Nick> text`."""
    return CHAT_LINE.format(nick=message.display_nick(nick_of), text=message.text)


def _bullets(items: Sequence[str]) -> list[str]:
    """Render items as bullets, or a placeholder when there are none."""
    return [STATE_BULLET.format(text=i) for i in items] or [STATE_BULLET.format(text=EMPTY)]


def render_debate_state(state: DebateState, nick_of: Callable[[str], str]) -> str:
    """Render the debate state as compact plain text (for prompts and `/state`)."""
    positions = [f"{nick_of(slug)}: {pos}" for slug, pos in state.positions.items()]
    disagreements = [
        f"{d.topic} — "
        + "; ".join(STATE_SIDE.format(nick=nick_of(s), stance=st) for s, st in d.sides.items())
        for d in state.disagreements
    ]
    lines = [STATE_USER_FACTS, *_bullets(state.user_facts)]
    lines += [STATE_POSITIONS, *_bullets(positions)]
    lines += [STATE_AGREEMENTS, *_bullets(state.agreements)]
    lines += [STATE_DISAGREEMENTS, *_bullets(disagreements)]
    lines += [STATE_OPEN_QUESTIONS, *_bullets(state.open_questions)]
    return "\n".join(lines)


def synthesis_vote(
    votes: Sequence[Vote], round_no: int, nick_of: Callable[[str], str]
) -> str | None:
    """Render the latest vote for the synthesis task; None if nobody voted."""
    lines = [
        SYNTHESIS_VOTE_LINE.format(
            nick=nick_of(v.speaker), confidence=v.confidence, position=v.position
        )
        for v in votes
        if v.error is None
    ]
    return SYNTHESIS_VOTE.format(round=round_no, lines="\n".join(lines)) if lines else None


def render_dossier(entries: Sequence[tuple[str, str, bool, str | None]]) -> str:
    """Render dossier files, given as (name, text, summarized, stats), in attach order."""
    parts = []
    for name, text, summarized, stats in entries:
        entry = (DOSSIER_ENTRY_SUMMARIZED if summarized else DOSSIER_ENTRY).format(
            name=name, text=text
        )
        parts.append(f"{entry}\n\n{stats}" if stats else entry)
    return "\n\n".join(parts)


def floor_line(nick: str, prompt: str) -> str:
    """Render the moderator handing a member the floor, without doubling the @nick."""
    lead = f"@{nick}".lower()
    text = prompt.strip()
    while text.lower().startswith(lead):
        text = text[len(lead) :].lstrip(" ,:")
    return FLOOR_LINE.format(nick=nick, prompt=text or prompt)


def render_research(entries: Sequence[tuple[str, str, str]]) -> str:
    """Render research notes, given as (focus, date, text), newest first."""
    return "\n\n".join(
        RESEARCH_ENTRY.format(focus=focus, date=date, text=text)
        for focus, date, text in reversed(entries)
    )


def render_memory(private: str, shared: str) -> str | None:
    """Render the memory section body, or None when there is no memory."""
    parts = []
    if private.strip():
        parts.append(MEMORY_PRIVATE.format(text=private.strip()))
    if shared.strip():
        parts.append(MEMORY_SHARED.format(text=shared.strip()))
    return "\n\n".join(parts) or None


def render_dynamics(
    stats: Sequence[MemberStats],
    nick_of: Callable[[str], str],
    stalled: int,
    stall_limit: int,
    flaws: Sequence[Flaw] = (),
) -> str:
    """Render the engine's debate dynamics for the moderator (stall line if limit > 0).

    `flaws`: the figures the engine flagged in the round's turns (M11).
    """
    lines = []
    for s in stats:
        facts = [DYNAMICS_TURNS.format(turns=s.turns)]
        facts.append(
            DYNAMICS_SPOKE
            if s.silent_rounds == 0
            else DYNAMICS_SILENT.format(count=s.silent_rounds)
        )
        if s.updated:
            facts.append(DYNAMICS_UPDATED)
        if s.repeat >= REPEAT_THRESHOLD:
            facts.append(DYNAMICS_REPEAT.format(percent=round(s.repeat * 100)))
        lines.append(DYNAMICS_MEMBER.format(nick=nick_of(s.slug), facts="; ".join(facts)))
    if stall_limit > 0:
        lines.append(DYNAMICS_STALL.format(count=stalled, limit=stall_limit))
    if flaws:
        lines += ["", DYNAMICS_FLAWS, *_flaw_lines(flaws, nick_of)]
    return section(TITLE_DYNAMICS, "\n".join(lines))


_DYNAMICS_LABELS = {
    FlawKind.MISLABELLED: DYNAMICS_MISLABELLED,
    FlawKind.WRONG_UNIT: DYNAMICS_WRONG_UNIT,
}


def _flaw_lines(flaws: Sequence[Flaw], nick_of: Callable[[str], str]) -> list[str]:
    """Render engine-flagged figures for the moderator, at most MAX_DYNAMICS_FLAWS."""
    lines = [
        DYNAMICS_FLAW.format(
            nick=nick_of(f.member),
            figure=f.note,
            excerpt=f.excerpt,
            label=_DYNAMICS_LABELS.get(f.kind, ""),
        )
        for f in flaws[:MAX_DYNAMICS_FLAWS]
    ]
    if len(flaws) > MAX_DYNAMICS_FLAWS:
        lines.append(DYNAMICS_FLAWS_MORE.format(count=len(flaws) - MAX_DYNAMICS_FLAWS))
    return lines


def flaw_rule(cap: int, last_round: bool) -> str:
    """Tell the moderator what happens to the flaws it reports."""
    if cap <= 0 or last_round:
        return FLAW_RULE_SYNTHESIS
    return FLAW_RULE_PROMPTED.format(cap=cap)


def flaw_turn_prompt(quote: str, note: str, lang: Language, then: str | None = None) -> str:
    """Render the floor prompt asking a member to check a flaw (plus its planned prompt)."""
    note = note.strip()
    if note and note[-1] not in ".!?":
        note += "."
    prompt = FLAW_TURN_PROMPT[lang].format(quote=quote, note=note).replace("  ", " ")
    return prompt + (FLAW_TURN_THEN[lang].format(prompt=then) if then else "")


def synthesis_errors(
    flaws: Sequence[Flaw], nick_of: Callable[[str], str], lang: Language
) -> str | None:
    """Render the moderator's flaw reports for the synthesis task; None if there are none."""
    lines = [
        SYNTHESIS_ERROR_LINE.format(
            round=f.round,
            nick=nick_of(f.member),
            kind=f.kind.value,
            quote=f.excerpt,
            note=f" — {f.note}" if f.note else "",
        )
        for f in flaws
    ]
    if not lines:
        return None
    return task(SYNTHESIS_ERRORS, lang, lines="\n".join(lines))


def summon_rule(candidates: Sequence[Persona]) -> str:
    """Pick the moderator's summon instruction, listing the personas it may propose."""
    if not candidates:
        return SUMMON_RULE_OFF
    available = "; ".join(
        SUMMON_CANDIDATE.format(
            slug=p.slug,
            name=p.name,
            role=p.meta.role.value,
            expertise=f"; {', '.join(p.meta.expertise)}" if p.meta.expertise else "",
        )
        for p in candidates
    )
    return SUMMON_RULE_ON.format(available=available)


def synthesis_asked(asked: Sequence[str]) -> str | None:
    """Return the synthesis's rule against re-asking; None if nothing was asked."""
    if not asked:
        return None
    return SYNTHESIS_ASKED.format(asked="; ".join(f'"{q}"' for q in asked))


def asked_rule(asked: Sequence[str]) -> str:
    """Return the moderator's rule against re-asking, listing the questions already asked."""
    if not asked:
        return ASKED_RULE_NONE
    return ASKED_RULE.format(asked="; ".join(f'"{q}"' for q in asked))


def stop_rule(auto_stop: bool, last_round: bool, first_round: bool = False) -> str:
    """Pick the moderator's stop instruction for this update."""
    if last_round:
        return STOP_RULE_LAST_ROUND
    if first_round:
        return STOP_RULE_FIRST_ROUND
    return STOP_RULE_AUTO if auto_stop else STOP_RULE_MANUAL


# --- Persona generator (M13): `council personas new "<description>"` -------------
# Not a debate prompt, so PROMPTS_VERSION (recorded per session) doesn't change.

PERSONA_DRAFT_SYSTEM = (
    "You write persona files for council, a local app where a few LLM personas debate a "
    "user's question under a moderator, IRC-style. Each persona's text becomes its system "
    "prompt, often run by a small local model, so it must be short, concrete and easy to "
    "follow. Return only the requested JSON."
)

PERSONA_DRAFT_TASK = """Draft a persona from this description:

<description>
{description}
</description>

Rules:
- name: one capitalized word of at most 16 letters, the nick shown in the chat. Prefer a trait \
or a trade (like Rigor, Fixer, Value, Stoic); use a real person's name only when the \
description asks for that person. Taken names: {names}.
- slug: the name in lowercase. Taken slugs: {slugs}.
- role: "advisor", unless the description asks for a devil's advocate ("devil_advocate", \
must challenge the leading position) or a risk officer ("risk_officer", must name the \
failure modes).
- color: a #rrggbb color readable on a dark background, not one of: {colors}.
- expertise: 2 to 4 short areas. tags: the councils it fits, from: {councils}; others are \
allowed when none fits.
- who: one or two sentences in the second person, starting "You are <name>, ".
- think: 3 to 5 bullets in the second person, on how the persona reasons. One bullet says \
how it handles a figure it wasn't given: it names the unknown in words or says it is \
unknown, and never makes up a number, share, probability or price.
- biases: 1 or 2 bullets naming a bias it owns, ending "Say so when relevant."
- speak: one or two sentences on its voice, ending with "Concise."
- Write in English, whatever the description's language. No real personal data.

Here is an existing persona file, for the style and length (yours is assembled from your \
fields into the same sections):

<example>
{example}
</example>"""


# --- Comparison judge (M14): `council compare ... --judge` ------------------------
# Not a debate prompt, so PROMPTS_VERSION (recorded per session) doesn't change.

COMPARE_JUDGE_SYSTEM = (
    "You compare the final syntheses of several runs of the same debate, held by "
    "different line-ups of LLM personas. Judge only what the texts show. You don't know "
    "which models ran which variant, and you must not guess. Be concise and concrete."
)

COMPARE_JUDGE_TASK = """The question debated:

<question>
{topic}
</question>

Facts every variant was given (from a web search), to check figures against:

<facts>
{facts}
</facts>

The syntheses, one per variant:

{syntheses}

Compare them in Markdown, with these sections:
## Verdict
One short paragraph: which variant served the question best, and why.
## Quality
One bullet per variant: how clear, complete and actionable its synthesis is.
## Errors
One bullet per variant: figures or claims that contradict the facts, or that look \
made up, quoted briefly. Say "none found" when there are none.
## Agreement
Where the variants agree, and where their recommendations differ.

Refer to the variants by their letters. Write in {language}."""

COMPARE_JUDGE_SYNTHESIS = """<variant letter="{letter}">
{text}
</variant>"""

COMPARE_JUDGE_NO_FACTS = "(no web research in this comparison)"
