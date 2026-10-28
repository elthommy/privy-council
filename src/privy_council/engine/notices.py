"""Chat lines the engine posts as `system` messages.

UI text, not prompt text: models never see them (prompts live in prompts.py).
"""

NOTICE_LANGUAGE = "Debate language: {name}. /lang {other} switches it before round 1"
NOTICE_LANGUAGE_CURRENT = "Debate language: {name}"
NOTICE_LANGUAGE_SET = "Debate language set to {name}"
NOTICE_LANGUAGE_LOCKED = "The debate language ({name}) can't change once members have spoken"
NOTICE_LANGUAGE_USAGE = "usage: /lang en | fr"
NOTICE_CUT_OFF = "{nick} was cut off (hit the output cap)"
NOTICE_NO_ANSWER = "{nick} couldn't answer: {error}"
NOTICE_OPENING_FAILED = "Opening failed ({error}); continuing without it"
NOTICE_MODERATOR_PROBLEM = "Moderator reply rejected, retrying: {problem}"
NOTICE_MODERATOR_FALLBACK = "Moderator failed twice; falling back to round-robin turns"
NOTICE_NEXT = "Next round: {speakers}. /step to continue, /close to wrap up"
NOTICE_ENDED = "Debate ended: {reason}. /step or /close for the synthesis, /more N adds rounds"
NOTICE_REACHED_MAX = "max rounds reached"
NOTICE_REASON_MODERATOR = "the moderator ended the debate"  # a stop with no reason given
NOTICE_ENDED_EARLY = (
    "Debate stopped early: {reason}. /close for the synthesis, or /step or /run N to keep debating"
)
NOTICE_REASON_LOOPING = "no new arguments in the last {count} rounds"
NOTICE_REASON_SETTLED = "the remaining disagreements wouldn't change the decision"
NOTICE_CONTINUED = "Debate continues. Next round: {speakers}"
NOTICE_SILENT_ADDED = "{nick} gets the floor too: quiet for {count} rounds"
NOTICE_FLAW_ANSWERED = "{nick}'s figure flagged as unsourced is in the new research; flag dropped"
NOTICE_SUMMON_PROPOSAL_IGNORED = "Moderator proposed summoning {who!r}, who can't join; ignored"
NOTICE_CLOSED = "Session closed; synthesis saved"
NOTICE_ALREADY_CLOSED = "Session is already closed"
NOTICE_FALLBACK_ON = "Claude limit reached: Claude personas now use the local fallback model"
NOTICE_RATE_LIMIT = (
    "Claude usage limit reached{when}. Session paused: /step retries, or set "
    "[claude] fallback_to_local = true"
)
NOTICE_STEP_FAILED = "{phase} failed ({error}); /step to retry"
NOTICE_RESEARCH_OFF = (
    "Research is off for this session (--no-research, or [research] enabled = false)"
)
NOTICE_LOCAL_ONLY = (
    "Local-only mode: every seat runs on a local model and nothing reaches Claude. "
    "No web research: facts come from you and the dossier, other figures from the "
    "members' memory"
)
NOTICE_RESEARCH_NEEDS_CLAUDE = (
    "Research needs a Claude-backed moderator (web search runs through the claude CLI)"
)
NOTICE_RESEARCH_SEARCHED = "{nick} searched the web: {queries}"
NOTICE_RESEARCH_NO_SEARCH = "{nick} answered without searching the web"
NOTICE_RESEARCH_FAILED = "Research failed ({error}); the debate continues without it"
# Shown with each research note, so the quality difference stays visible.
RESEARCH_SOURCE_LABELS = {"claude": "Claude web search", "local": "local search"}
NOTICE_LOCAL_ONLY_SEARCH = (
    "Local-only mode: every seat runs on a local model and nothing reaches Claude. "
    "Web search via SearXNG: search queries leave this PC"
)
# First local search of a normal session ([research] provider = "local" or "auto").
NOTICE_LOCAL_SEARCH = "Web search via SearXNG: search queries leave this PC"
NOTICE_RESEARCH_NEEDS_LOCAL_SEARCH = (
    "Research in local-only mode needs local search: set [research.local] enabled = true "
    "and start SearXNG (scripts/searxng.sh start)"
)
NOTICE_LOCAL_SEARCHED = "{nick} searched the web (local search): {queries} ({count} result(s))"
NOTICE_LOCAL_QUERY_FAILED = "Local search failed for {query!r} ({error})"
NOTICE_LOCAL_FOLLOW_UP = "{nick} searches again for what is missing: {facts}"
NOTICE_LOCAL_RETRY = "No result: {nick} searches again with shorter queries: {queries}"
NOTICE_LOCAL_NOTHING = "{nick}'s search found nothing ({queries}); the debate continues without it"
NOTICE_LOCAL_QUERIES_FALLBACK = (
    "{nick} gave no usable search queries ({problem}); searching the focus instead"
)
NOTICE_LOCAL_DROPPED = (
    "{count} research bullet(s) dropped: their figures or URLs are not in the search results"
)
NOTICE_SYNTHESIS_MODEL_LIMITED = (
    "{model} hit a usage limit; writing the synthesis with {fallback} instead"
)
NOTICE_PAUSING = "Pausing after the current turn…"
NOTICE_PAUSED = "Paused. /step or /run N to continue"
NOTICE_RUN_AT_END = "The debate has ended: /close for the synthesis, or /more N to add rounds"
NOTICE_MORE = "{count} more round(s): up to round {limit}. /step or /run N to continue"
NOTICE_BUSY = "Busy: wait for the current turn, or /pause"
NOTICE_UNKNOWN_MEMBER = "No member called {who!r} in this channel"
NOTICE_MEMORY_SAVED = "{nick} saved {count} memory note(s)"
NOTICE_MEMORY_NOTHING = "{nick} had nothing new to remember"
NOTICE_MEMORY_SILENT = "{nick} didn't speak this session; no notes written"
NOTICE_MEMORY_READ_ONLY = "Memory is read-only for this council: nothing saved"
NOTICE_MEMORY_OFF = "Memory is off for this council: nothing saved"
NOTICE_MEMORY_FAILED = "{nick} couldn't write memory ({error}); skipped"
NOTICE_MEMORY_MERGED = "{nick}'s notes passed {cap} bytes and were merged (old copy: notes.md.bak)"
NOTICE_MEMORY_MERGE_BAD = (
    "{nick}'s notes are over {cap} bytes, but the merge was unusable; kept as is"
)
NOTICE_SHARED_FAILED = "Shared memory proposal failed ({error}); nothing saved"
NOTICE_SHARED_UNQUOTED = "Dropped {count} proposed fact(s) not backed by your own words"
NOTICE_SHARED_NONE = "No new facts about you to remember"
NOTICE_SHARED_SAVED = "Saved {count} fact(s) to shared memory"
NOTICE_SHARED_NONE_KEPT = "No facts saved"
NOTICE_SHARED_MERGE_BAD = "Shared memory is over {cap} bytes, but the merge was unusable"
NOTICE_SHARED_REWRITTEN = "Shared memory replaced by the merged version (old copy: user.md.bak)"
NOTICE_SHARED_KEPT = "Shared memory kept as is"
NOTICE_KEEP_NOTHING = "Nothing awaits approval"
NOTICE_KEEP_USAGE = "usage: /keep 1 3 | all | none"
NOTICE_KEEP_REWRITE_USAGE = "usage: /keep all (use the merged version) | /keep none"
NOTICE_FORGET_USAGE = "usage: /forget @nick | shared | all"
NOTICE_FORGET_CONFIRM = "This deletes {what}. Type /forget {arg} again to confirm"
NOTICE_FORGOT = "Forgot {what}"
NOTICE_FORGET_EMPTY = "Nothing to forget: {what} was already empty"
FORGET_SHARED = "the shared facts about you"
FORGET_ALL = "all memory (every persona's notes and the shared facts)"
FORGET_PERSONA = "{nick}'s private notes"
NOTICE_AUTO_RESEARCH = "{nick} is looking up: {focus}"
NOTICE_NEEDS_TOPIC = "No question yet: type it to start the session"
NOTICE_TOPIC_SET = "Question set; opening the debate"
NOTICE_QUESTIONS_PAUSE = "/run paused for {nick}'s question"
ASK_LINE = "@{nick} {question}"
ROUND_LABEL_BLIND = "Round 1 — blind takes"
ROUND_LABEL_OPEN = "Round {round} — open takes"
ROUND_LABEL_CROSS = "Round {round} — cross-examination"

# Roster (engine/roster_flow.py)
NOTICE_JOINED = "{nick} joined the channel and speaks next round"
NOTICE_JOINED_LATE = "{nick} joined the channel; the debate has ended, so /ask @{nick} or /close"
NOTICE_LEFT = "{nick} left the channel"
NOTICE_SUMMON_USAGE = "usage: /summon <persona> (council personas list)"
NOTICE_SUMMON_UNKNOWN = "No persona called {who!r}; available: {available}"
NOTICE_SUMMON_MEMBER = "{nick} is already in the channel"
NOTICE_SUMMON_MODERATOR = "{nick} is a moderator persona and can't join as a member"
NOTICE_KICK_USAGE = "usage: /kick <nick>"
NOTICE_KICK_MODERATOR = "{nick} moderates this session and can't be kicked"
NOTICE_KICK_LAST = "The channel needs at least one member"
NOTICE_SESSION_CLOSED = "The session is closed"

# Resume (engine/resume.py)
NOTICE_RESUMED_OPENING = "Resumed: the opening hasn't run yet. /step starts it"
NOTICE_RESUMED_MID_ROUND = "Resumed in round {round}: {count} turn(s) still to run. /step continues"
NOTICE_RESUMED_AFTER_ROUND = "Resumed after round {round}. /step runs the next one, /close wraps up"
NOTICE_RESUMED_SYNTHESIS = "Resumed: the debate has ended. /step or /close writes the synthesis"
NOTICE_RESUMED_EARLY_STOP = (
    "Resumed: the debate stopped early. /close writes the synthesis, /step keeps debating"
)
NOTICE_RESUMED_ERROR = "Resumed after an error. /step retries"
NOTICE_RESUMED_CLOSED = "Resumed a closed session: shared facts still await /keep"
NOTICE_RESUMED_CLOSING = (
    "Resumed: the app stopped while writing memory notes, so some may be missing. "
    "Session closed; synthesis saved"
)

# Taking stock (engine/review_flow.py)
NOTICE_NOTHING_YET = "Nothing to {what} yet: /step runs round 1 first"
NOTICE_VOTE_INVALID = "unusable vote ({problem})"
NOTICE_SUMMARY_DONE = "Interim summary; the debate continues: /step or /run N"
NOTICE_SUMMARY_FAILED = "Summary failed ({error}); /summary to retry"

# Dossier (engine/dossier_flow.py)
NOTICE_ATTACH_USAGE = "usage: /attach <path> ({allowed} files)"
NOTICE_ATTACH_MISSING = "No file at {path}"
NOTICE_ATTACH_TYPE = "Can't attach {name}: only {allowed} files are allowed"
NOTICE_ATTACH_TOO_BIG = (
    "{name} is {size:,} bytes; the limit is {limit:,} ([dossier] max_file_bytes)"
)
NOTICE_ATTACH_NOT_TEXT = "Can't attach {name}: it isn't UTF-8 text"
NOTICE_ATTACH_EMPTY = "Can't attach {name}: it is empty"
NOTICE_ATTACHED = "Attached {name} (~{tokens:,} tokens): everyone sees it from the next turn"
NOTICE_ATTACHED_SUMMARIZED = (
    "Attached {name} (~{tokens:,} tokens, too long to include whole): everyone sees "
    "{nick}'s summary from the next turn"
)
NOTICE_SUMMARIZING = "{nick} is summarizing {name} (~{tokens:,} tokens)…"
NOTICE_SUMMARIZE_CLIPPED = (
    "{name} is too long to read at once: only its first {percent}% is summarized"
)
NOTICE_ATTACH_FAILED = "Couldn't summarize {name} ({error}); nothing attached. /attach to retry"
