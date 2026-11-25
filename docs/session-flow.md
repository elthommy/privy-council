# Session flow

Three diagrams, from far to near: the whole session, one round, one member's turn.
The steps marked *forecast* run only in councils with `estimate: probability`
(economics, geopolitics). "Engine" is plain Python: it calls no model itself, it
decides who is called and checks what comes back. Budget checks, `/summon`, pausing
and the Discord front end are left out.

## 1. The whole session

```mermaid
flowchart TD
    Q([You ask a question]) --> S1["Chair searches the web<br/>background, your premise,<br/>the measure, the cause of a market move"]
    S1 --> O["Chair opens<br/>restates the question,<br/>asks you 1-2 questions on its terms"]
    O --> A1([You answer])
    A1 --> S2["Chair searches again<br/>where your terms stand today<br/>(forecast)"]
    S2 --> R1["Round 1, blind<br/>members don't see each other;<br/>devil's advocate last, sees them"]
    R1 --> U["Chair updates<br/>diagram 2"]
    U -->|debate goes on| RN["Next round<br/>members answer the Chair's<br/>prompts and each other"]
    RN --> U
    U -->|"Chair says consensus,<br/>2 rounds without progress,<br/>or last round"| E["Debate ends"]
    E --> CL(["/close"])
    CL --> L["Late-facts turn (forecast)<br/>members whose number is older<br/>than the newest facts speak once more"]
    L --> SY["Chair writes the synthesis<br/>agreements, disagreements,<br/>median estimate, errors, next steps"]
    SY --> M["Memory<br/>each member writes notes, Chair checks them;<br/>facts about you need your OK"]
    M --> Z([Closed])
```

## 2. One round (2 and later)

```mermaid
sequenceDiagram
    autonumber
    actor You
    participant Chair
    participant Engine
    participant Web as Web / page reader
    participant Seats as Members

    rect rgba(120,150,200,0.15)
    note over You,Seats: Before the round: did you answer a question? (forecast)
    Engine->>Chair: Does this answer change what counts?
    opt it does
        Chair->>Web: One search on what changed
        Web-->>Engine: New facts, numbered [F32], [F33]...
    end
    end

    rect rgba(200,170,100,0.15)
    note over You,Seats: The round
    loop each planned member, one at a time
        Engine->>Seats: Chair's prompt + your answers since<br/>+ facts + debate state + last messages
        Seats-->>Engine: Answer, with [F] tags and an estimate
        Engine->>Engine: Check the answer (diagram 3)
    end
    end

    rect rgba(120,190,140,0.15)
    note over You,Seats: The Chair's update (one JSON reply)
    Engine->>Chair: This round's answers + engine flags<br/>+ estimates + facts
    Chair-->>Engine: Debate state, errors found, next prompts,<br/>searches, page to read, questions for you, stop?
    Engine->>Engine: Drop flags that don't hold,<br/>put the first 2 in front of next prompts
    Engine->>Engine: Progress? New arguments that matter,<br/>or an estimate that moved 3+ points
    end

    alt debate ends
        Engine-->>You: "Debate stopped", then /close
    else debate goes on
        opt Chair asked for lookups
            Engine->>Web: Up to 2 searches + 1 page<br/>(a page it can't read becomes a search)
            Web-->>Engine: New facts, for the next round
        end
        opt Chair has questions
            Engine-->>You: Question (/run pauses)
            You->>Engine: Your answer, used in the next round
        end
    end
```

The engine ends a debate after `stall_rounds` (2) rounds without progress, but gives it
one more round if the Chair had just asked for a lookup that would really run.

## 3. One member's turn

```mermaid
flowchart TD
    P["Engine builds the prompt<br/>local models: short links, and the facts<br/>the Chair asked about come first"] --> D["Member drafts"]
    D --> K{"Local model with repair on:<br/>engine finds a format, tag or<br/>arithmetic problem?"}
    K -->|no, or a Claude member| POST["Posted in the chat"]
    K -->|yes| F["Same member rewrites once"]
    F -->|fewer problems| POST
    F -->|no better| D2["The draft is kept"] --> POST
    POST --> C["Engine checks it and logs flags"]
    C --> EST["Engine reads the estimate line<br/>(forecast)"]
    EST --> UP["Waits for the Chair's update"]
```

## Error checks

What "flag errors" does above. The engine's checks are plain Python
(`engine/checks.py`, no model call). They only test where a figure comes from:
whether it is in the member's own prompt or the member's own arithmetic on figures
from the same turn. Wrong sums, contradictions and logic errors are judged by the
Chair alone. The engine then drops the Chair's reports that don't hold up. A posted
turn is never changed; only a local member's unposted draft can be (diagram 3).

```mermaid
flowchart TD
    A[Member writes a turn] --> B["Engine: check_turn<br/>(engine/checks.py, no model)"]
    B -->|"figure found nowhere in the member's prompt<br/>→ unsourced / wrong_unit / mislabelled"| C[Engine flags]
    B -->|"format slips<br/>(Nick: prefix, bad @mention…)"| L[Logged only]

    C -->|"listed in 'Debate dynamics'<br/>(max 8)"| D["Chair reviews the round<br/>returns JSON flaws (max 4):<br/>arithmetic · contradiction · logic · unsourced · other"]

    D --> E{"Engine cross-check<br/>(engine/flaw_flow.py)"}
    E -->|"figure is in the member's sources,<br/>or the member's own arithmetic,<br/>or the Chair's note says it holds"| X[Flag dropped]
    E -->|holds| F[Flaw recorded]
    F -->|"new research gives the figure"| X

    F -->|"first max_flaw_prompts_per_round (2)"| G["Next round: Chair asks the member<br/>'you wrote …, check it'"]
    F -->|the rest| H[Final synthesis]
    G -->|"member answers"| OK[Resolved]
    G -->|"no answer"| H
    H --> S["'Errors' section<br/>in the synthesis"]
```

`/flaws` lists every flag, from both the engine and the Chair.
