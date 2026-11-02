# Session flow

A standard council session, at a high level. Pause/resume, memory, the budget
checks and `/summon` are left out.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Chair as Chair (moderator)
    participant Web as Web research
    participant Members as Council members

    User->>Chair: Ask a question (+ optional dossier)

    rect rgba(120,150,200,0.15)
    note over Chair,Web: Opening
    Chair->>Web: Search for background facts
    Web-->>Chair: Sourced results
    Chair->>Chair: Reformulate the question,<br/>build debate state and turn plan
    Chair-->>User: Reframed question (+ clarifying questions)
    end

    rect rgba(200,170,100,0.15)
    note over Chair,Members: Round 1: blind
    loop each member, independently
        Chair->>Members: Question + debate state (no other answers)
        Members-->>Chair: Initial take
    end
    note right of Members: Devil's advocate goes last<br/>and sees the blind takes
    Chair->>Chair: Update debate state, flag errors
    end

    loop Rounds 2..N
        Chair->>Members: Planned turns + debate state + last messages
        Members-->>Chair: Rebuttals and refinements
        Chair->>Chair: Update state, flag errors, plan next round
        opt Chair needs input
            Chair-->>User: Question
            User->>Chair: Answer
        end
    end

    Chair->>Members: /vote
    Members-->>Chair: Positions and confidence
    Chair-->>User: Synthesis (recommendation, errors caught, next steps)
```

## Error checks

What "flag errors" does above. The engine's checks are plain Python
(`engine/checks.py`, no model call). They only test where a figure comes from:
whether it is in the member's own prompt or the member's own arithmetic on figures
from the same turn. Wrong sums, contradictions and logic errors are judged by the
Chair alone. The engine then drops the Chair's reports that don't hold up. A
flagged turn's text is never changed.

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
