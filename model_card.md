# AI Interactions Log

> **Stretch features only.** Only fill in the sections that apply to stretch features you attempted. If you did not attempt a stretch feature, leave its section blank or delete it. This file is not required for the core project.

---

## Agentic Workflow (SF8)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**

After implementing the core `score_song` and `recommend_songs` functions,
I used ChatGPT as a **design reviewer** rather than a code generator. The
task was to stress-test the scoring function against adversarial inputs
and identify failure modes that I had not documented in the Model Card.

**Prompts used:**

I pasted the full `score_song` function and asked:

> *"What inputs or user profiles would make this score behave badly?
> List at least 3 failure modes. For each one, tell me if it's fixable
> with a small code change (under 10 lines)."*

This adversarial framing is deliberate. My earlier workflow was to ask
the AI to **write** code (which it did well but without questioning
design assumptions). This time I asked it to **critique** code — and the
quality of the response was dramatically different.

**What did the agent generate or change?**

ChatGPT produced **7 failure modes**, each with a concrete example input
and a proposed fix:

| # | Failure mode | Fixable in <10 LOC? |
|---|---|---|
| 1 | **Popularity bias** — fixed 0.5 bonus can override categorical preferences | Yes (partial) |
| 2 | **Energy mismatch** — assumes both values are in `[0, 1]`; out-of-range inputs produce negative contributions | Yes |
| 3 | **Invalid numeric values** — no validation; `None` or strings cause crashes or nonsensical scores | Yes |
| 4 | **Unbounded tempo penalty** — linear with no cap; a song at 1000 BPM subtracts 8.8 points | Yes |
| 5 | **Hard acoustic threshold** — `acousticness > 0.6` treats 0.599 and 0.601 as completely different | Yes |
| 6 | **Missing preferences create accidental matches** — if both `favorite_genre` and `song["genre"]` are `None`, the equality test succeeds and awards points | Yes |
| 7 | **Poorly calibrated weights** — if one weight dominates, the recommender ignores other signals | Sometimes |

For #4, ChatGPT suggested capping the penalty:

```python
score -= min((song["tempo_bpm"] - max_tempo) / 100, 1.0)
```

For #6, it suggested guarding the comparison:

```python
genre = user_prefs.get("favorite_genre")
if genre is not None and song.get("genre") == genre:
    score += weights["genre"]
```

**What did you verify or fix manually?**

I cross-referenced ChatGPT's list against my Model Card:

- **#5 (acoustic threshold)** was already documented in Model Card §5g.
- **#1 (popularity bias)** was already documented in Model Card §5c.
- **#2, #3, #4, #6, #7** were **new findings** I had not documented.

I decided not to patch all of them into `recommender.py` because:

1. The **core project rubric** does not require input validation — the
   CSV is hand-authored and controlled, so invalid ranges cannot occur
   in practice.
2. Patching #4 (tempo cap) and #6 (preference guard) would change the
   scoring formula, which would invalidate the 28 existing unit tests
   and the `output.txt` snapshot.
3. The **deeper limitations** — hard thresholds, fixed weights, no
   feedback loop — are inherent to the rule-based design, not bugs.
   Fixing them would require rethinking the architecture (see Model
   Card §7).

The value of this exercise was not the patches themselves, but the
**confirmation** that the scoring function has well-understood failure
modes. ChatGPT's closing line captured it exactly:

> *"Most of these problems are fixable with small code changes. The
> deeper limitations are hard thresholds, fixed weights, and the
> absence of feedback from actual user behavior."*

**Takeaway:** AI as a *code generator* produces working implementations
but does not volunteer risks. AI as a *design reviewer* — when prompted
adversarially — surfaces failure modes I would have missed. The prompt
matters more than the model.

---

## Design Pattern (SF10)

> Document how AI helped you choose or implement a design pattern.

**Which design pattern did you use?**

**Strategy pattern**, implemented via a **data-driven weight table**
dictionary rather than a class hierarchy.

**How did AI help you brainstorm or implement it?**

After implementing the three ranking modes, I asked Gemini to validate
my design choice:

> *"Is this a valid implementation of the Strategy pattern even though
> I'm not using classes? When should I refactor this into a class-based
> Strategy pattern?"*

Gemini confirmed that the dictionary approach **is** a valid
implementation — specifically a *"data-driven or functional variation"*
— because it satisfies the core intent of the pattern:

> *"The core intent of the Strategy pattern is to define a family of
> interchangeable algorithms or behaviors and let a client select them
> dynamically at runtime. By storing your ranking weights in a
> dictionary and selecting them via a key lookup, you achieve complete
> decoupling of the selection mechanism from the execution logic."*

It then gave **two specific triggers** that would justify refactoring
to a class hierarchy:

1. **Complex algorithmic logic** — if a mode requires conditional
   steps, normalization, or external data lookups unique to it, a data
   dictionary no longer suffices.
2. **Strict interface enforcement** — if the application grows and
   needs to guarantee that every strategy implements specific methods
   or undergoes type-checking, an abstract base class provides the
   necessary contract.

Neither trigger applies to this project: all three modes differ only in
numerical weights, and there are no shared methods to enforce. The
dictionary approach is therefore the correct design for the current
scope, and this decision is recorded in **ADR-003**.

**How does the pattern appear in your final code?**

In `src/recommender.py`:

```python
RANKING_MODES = {
    "genre-first":       {"genre": 3.0, "mood": 2.0, "energy": 2.0, "acoustic": 1.0},
    "mood-first":        {"genre": 2.0, "mood": 3.0, "energy": 2.0, "acoustic": 1.0},
    "energy-similarity": {"genre": 1.0, "mood": 1.0, "energy": 4.0, "acoustic": 1.0},
}

DEFAULT_MODE = "genre-first"

def score_song(user_prefs, song, mode=DEFAULT_MODE):
    weights = RANKING_MODES.get(mode, RANKING_MODES[DEFAULT_MODE])
    # ... use weights["genre"], weights["mood"], etc.
```

The **context** (`score_song`) is decoupled from the **concrete
strategy** (the weight table), and the caller selects the strategy by
passing `mode=...`. Adding a fourth mode is a single dictionary entry —
no new class, no new function, no changes to callers. This decision is
documented in **ADR-003** in `ARCHITECTURE.md`.