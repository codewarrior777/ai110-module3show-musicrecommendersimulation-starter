# Architecture Decision Records (ADRs) — Music Recommender

This document captures the key architectural decisions made during the
design and implementation of the Music Recommender Simulation.

---

## ADR-001: Two parallel APIs (functional + OOP)

**Status:** Accepted

**Context:**
The starter template provided two separate APIs for the same logic:
- A **functional API** (`load_songs`, `score_song`, `recommend_songs`)
  used by `src/main.py` and the CLI.
- An **OOP API** (`Recommender` class with `Song` and `UserProfile`
  dataclasses) used by `tests/test_recommender.py`.

Both must produce the same results.

**Decision:**
Implement the functional API as the **single source of truth**, and make
the OOP API a **thin wrapper** that converts dataclasses to dicts, calls
the functional scorer, and maps results back to `Song` objects.

**Rationale:**
- Duplicating scoring logic in two places invites drift between them.
- A wrapper keeps the "shape" of each API intact (so both tests and the
  CLI work) while guaranteeing they can never disagree.
- The conversion overhead (`_to_dict`) is O(1) per song — negligible.

**Consequences:**
- ✅ Single place to fix scoring bugs.
- ✅ OOP tests and functional CLI always stay consistent.
- ⚠️ The wrapper loses access to dataclass-only fields (e.g., it defaults
  `popularity` to 50 since `Song` doesn't carry that field).

---

## ADR-002: Explicit feature schema over learned embeddings

**Status:** Accepted

**Context:**
Real recommender systems (Spotify, YouTube Music) learn user and item
embeddings from listening behavior. We have neither the data volume nor
the training budget for that.

**Decision:**
Use a **fixed, explicit feature schema** for every song (15 attributes:
genre, mood, energy, tempo_bpm, valence, danceability, acousticness, plus
the 5 stretch attributes) and a **weighted linear scoring function**.

**Rationale:**
- With a 20-song catalog, a learned model would overfit immediately.
- Explicitness is a **feature**, not a limitation: every score is
  traceable to specific criteria, which is required by the rubric
  ("explanations for each recommended song").
- The scoring weights (`RANKING_MODES`) are editable by hand and readable
  by a non-engineer.

**Consequences:**
- ✅ Every recommendation comes with a natural-language explanation.
- ✅ Zero training cost, zero inference cost.
- ⚠️ The model cannot learn taste patterns it wasn't explicitly told about.

---

## ADR-003: Three ranking modes instead of one global formula

**Status:** Accepted

**Context:**
A single scoring formula forces one definition of "best song" on every
user. But an EDM fan and an acoustic study-session listener weigh
features very differently.

**Decision:**
Ship **three named ranking modes**, each with its own weight table:

| Mode | genre | mood | energy | acoustic |
|---|---|---|---|---|
| `genre-first` | 3.0 | 2.0 | 2.0 | 1.0 |
| `mood-first` | 2.0 | 3.0 | 2.0 | 1.0 |
| `energy-similarity` | 1.0 | 1.0 | **4.0** | 1.0 |

**Rationale:**
- The rubric's stretch feature #3 explicitly asks for ≥2 ranking modes.
- Weight tables are the simplest implementation of the Strategy pattern
  that doesn't require new classes.
- The mode name is a keyword argument, so the API stays stable when we
  add a 4th mode later.

**Consequences:**
- ✅ Same input can produce different (still defensible) orderings.
- ✅ Adding a new mode is a **single dictionary entry**.
- ⚠️ Users must choose a mode — no automatic mode selection yet.

---

## ADR-004: Diversity penalty at ranking time, not in the score

**Status:** Accepted

**Context:**
With 20 songs and only 4 artists repeated across multiple genres, a pure
top-K selection can return "After Hours → Sunrise City → Night Drive
Loop" where 2 of the 3 are by the same artist (Neon Echo). This is a
classic filter-bubble failure.

**Decision:**
Keep the **scoring function pure** (no diversity logic). Apply the
diversity penalty **only in `recommend_songs()`**, at selection time:

```python
adjusted_score = raw_score - (artist_repeats + genre_repeats) * 0.8