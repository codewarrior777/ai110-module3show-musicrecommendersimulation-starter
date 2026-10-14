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

Both must use the same scoring logic.

**Decision:**
Implement the functional API as the **single source of truth**, and make
the OOP API a **thin wrapper** that converts dataclasses to dicts, calls
the functional scorer, and maps results back to `Song` objects. The two
APIs share one scoring formula, so their *logic* cannot drift.

**Rationale:**
- Duplicating scoring logic in two places invites drift between them.
- A wrapper keeps the "shape" of each API intact (so both tests and the
  CLI work) while guaranteeing they always use the same formula.
- The conversion overhead (`_to_dict`) is O(1) per song — negligible.

**Consequences:**
- ✅ Single place to fix scoring bugs.
- ✅ The scoring *formula* is shared, so the two APIs cannot drift in
  their logic.
- ✅ OOP tests and functional CLI stay aligned on genre, mood, energy,
  acoustic, and tempo signals.
- ⚠️ The two APIs are **not score-equivalent** for the popularity term:
  `_to_dict()` hardcodes `popularity=50` (and `release_year=2023`,
  `instrumentalness=0.0`, `liveness=0.0`) because the `Song` dataclass
  doesn't carry those fields. The OOP path therefore always applies a
  fixed `+0.25` popularity bonus, while the functional path uses the
  real CSV value. See Model Card §5f.
- ⚠️ Any future signal added to the functional scorer that depends on a
  field missing from `Song` will silently behave differently in the two
  APIs. The wrapper reduces drift, but does not eliminate it.

---

## ADR-002: Explicit feature schema over learned embeddings

**Status:** Accepted

**Context:**
Real recommender systems (Spotify, YouTube Music) learn user and item
embeddings from listening behavior. We have neither the data volume nor
the training budget for that.

**Decision:**
Use a **fixed, explicit feature schema** for every song (15 attributes:
`id`, `title`, `artist`, `genre`, `mood`, `energy`, `tempo_bpm`,
`valence`, `danceability`, `acousticness`, plus the 5 stretch attributes
`popularity`, `release_year`, `detailed_mood_tags`, `instrumentalness`,
`liveness`) and a **weighted linear scoring function**.

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
With 20 songs and **8 artists repeated across the catalog**, a pure top-K
selection can return consecutive tracks by the same artist (e.g., *Night
Drive Loop* and *After Hours* for a synthwave fan). This is a classic
filter-bubble failure. Four artists appear 3 times each (Neon Echo,
LoRoom, Slow Stereo, Indigo Parade), and four more appear twice (Voltline,
Orbit Bloom, Paper Lanterns, Max Pulse).

**Decision:**
Keep the **scoring function pure** (no diversity logic). Apply the
diversity penalty **only in `recommend_songs()`**, at selection time:

```python
artist_repeat = seen_artists.get(song["artist"], 0)
genre_repeat  = seen_genres.get(song["genre"], 0)
penalty       = (artist_repeat + genre_repeat) * diversity_weight  # default 0.8
adjusted_score = round(raw_score - penalty, 3)
```

The selection is **greedy**: candidates are pre-sorted by raw score,
picked one at a time while tracking seen artists/genres, and the final
list is **re-sorted by adjusted score** before returning.

**Rationale:**
- Keeping `score_song` pure means it stays unit-testable in isolation
  (no hidden state, deterministic output).
- The diversity penalty is a **ranking-time concern**, not a
  **feature of the song** — putting it inside `score_song` would require
  passing the already-picked set on every call.
- A fixed penalty of 0.8 per repeat is a reasonable default: it's smaller
  than a genre match (3.0) but larger than the popularity bonus (0.5), so
  it can flip ties without overriding strong signals.

**Consequences:**
- ✅ `score_song` remains pure and easy to test (28 tests, 100% coverage).
- ✅ Diversity behavior is visible in the output via
  `[diversity penalty: -0.80]` tags.
- ⚠️ The penalty is fixed at 0.8 regardless of how strong the underlying
  signal is. A per-user tunable would be the next step.

---

## ADR-005: CSV as the single source of truth for the catalog

**Status:** Accepted

**Context:**
The catalog needs to be editable by hand (adding songs, tweaking
features) without touching code, and readable by both the CLI and the
FastAPI layer.

**Decision:**
Store the catalog in `data/songs.csv` with a fixed header of 15 columns
and load it via `load_songs()` with explicit type conversion per column.

**Rationale:**
- CSV is diff-friendly, human-editable, and free of external dependencies.
- A single loader function (`load_songs`) means the CLI, the API, and
  the tests all read the same file the same way.
- Explicit type conversion (`int(row["id"])`, `float(row["energy"])`, …)
  catches schema drift at load time instead of at score time.

**Consequences:**
- ✅ Adding a song is a one-line CSV edit — no code change.
- ✅ All three entry points (CLI, API, tests) share one loader.
- ⚠️ Missing columns raise `KeyError` at load time with no friendly
  message. A schema-validation step would be a future improvement.
- ⚠️ At 20 rows, the catalog fits in memory trivially. A larger catalog
  (1000+) would want an index or database.

---

## ADR-006: FastAPI as a thin adapter over the domain layer

**Status:** Accepted

**Context:**
The project needs a REST API to demonstrate the recommender beyond the
CLI, but adding a second entry point risks duplicating business logic.

**Decision:**
Implement `src/api.py` as a **thin adapter**: Pydantic models for
request/response validation, and route handlers that delegate directly
to `load_songs` / `score_song` / `recommend_songs` from
`src/recommender.py`. No scoring logic lives in the API layer.

**Rationale:**
- The domain layer is the single source of truth; the API is a
  presentation concern.
- Pydantic gives us request validation and auto-generated Swagger docs
  for free at `/docs`.
- A thin adapter is easy to test — most logic is already covered by the
  domain tests, so the API layer needs only smoke tests.

**Consequences:**
- ✅ Swagger UI at `/docs` is auto-generated from the Pydantic schemas.
- ✅ The API cannot drift from the CLI — both call the same functions.
- ⚠️ The API layer is excluded from coverage via `.coveragerc`. A bug in
  a route handler (e.g., wrong status code, missing param) would not be
  caught by the current test suite.
- ⚠️ CORS is not configured. Fine for local development, not for a
  production deployment.