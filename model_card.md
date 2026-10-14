# Music Recommender — Model Card

**Model name:** Music Recommender Simulation v1.0
**Author:** Gustavo Ramos
**Course:** AI 110 — Foundations of AI Engineering
**Project:** Project 3 (Music Recommender Simulation)
**Date:** October 2025

---

## 1. Intended Use

**Intended use:** Recommend songs from a fixed 20-track catalog to a
user who has explicitly stated their preferences for genre, mood,
target energy level, and acoustic preference.

**Out-of-scope use:** The model is **not** intended for real production
recommendation at scale, for users whose taste cannot be expressed by the
four preference fields, or for any decision that affects a person's access
to content. It is a learning artifact, not a product.

---

## 2. Dataset Description

| Property | Value |
|---|---|
| **Size** | 20 songs |
| **Features per song** | 15 (id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness, popularity, release_year, detailed_mood_tags, instrumentalness, liveness) |
| **Genres covered** | pop, lofi, rock, ambient, jazz, synthwave, indie pop, folk |
| **Moods covered** | happy, chill, intense, relaxed, focused, moody, angry |
| **Release years** | 2019-2024 |
| **Source** | Hand-authored by the student for this project |
| **Data types** | 3 categorical (genre, mood, detailed_mood_tags) + 12 numeric |

The 5 extra attributes (popularity, release_year, detailed_mood_tags,
instrumentalness, liveness) were added as **stretch feature #1**.

---

## 3. Algorithmic Approach (plain language)

For each candidate song, the model produces a **score** by adding together
several signals:

1. **+3.0** if the song's genre matches the user's favorite genre.
2. **+2.0** if the song's mood matches the user's favorite mood.
3. **+ up to 2.0** for how close the song's energy is to the user's target
   energy. A perfect match gets 2.0; a maximum mismatch gets 0.0.
4. **+1.0** if the acoustic preference *fits* the song, where fit is defined
   as `likes_acoustic == (song.acousticness > 0.6)`. This is a **binary
   fit**, not a continuous similarity: a user who dislikes acoustic music
   receives the bonus for non-acoustic songs.
5. **+ up to 0.5** as a popularity micro-bonus (popularity / 100 × 0.5).
6. **− linear tempo penalty** `(tempo_bpm − max_tempo) / 100`, applied
   **only when** the user sets `max_tempo` and the song exceeds it. The
   three built-in profiles do not set `max_tempo`, so this term is 0.0 by
   default.

The **weights change by mode**:

- `genre-first` → genre counts the most (3.0).
- `mood-first` → mood counts the most (3.0).
- `energy-similarity` → energy counts the most (up to 4.0).

After scoring, the top-K songs are picked. **A diversity penalty of 0.8
per repeated artist or genre** is applied during a **greedy top-K
selection** (candidates are pre-sorted by raw score; the final list is
re-sorted by adjusted score). This prevents the list from collapsing into
"five songs by the same artist".

---

## 4. Evaluation

The model was evaluated by running 3 user profiles across 3 ranking modes
(9 combinations). Full output is saved in `output.txt`.

**Evaluation method:** Manual inspection of the top-5 per profile. No
held-out test set — the model is rule-based, so there is nothing to
"train" and nothing to overfit. Accuracy is not measured because there
is no ground-truth label; the deliverable is **interpretability of the
ranking**, not predictive accuracy.

**Sanity checks performed:**

- ✅ EDM profile surfaces high-energy pop (Gym Hero, Rooftop Reprise)
- ✅ Chill profile surfaces lofi with high acousticness (Library Rain, Focus Flow)
- ✅ Synthwave profile surfaces moody synthwave (After Hours, Night Drive Loop)
- ✅ Diversity penalty visibly kicks in when the same artist repeats
- ✅ All 3 modes produce different orderings on the same profile

**Automated verification:** 28 unit tests in `tests/test_recommender.py`
pass with **100% coverage** on `src/recommender.py` (verified via
`pytest --cov`). These tests do not measure ranking quality — they
measure that the scoring and selection logic behaves as specified.

---

## 5. Limitations and Biases

### 5a. Genre imbalance

The dataset is distributed as **4 pop, 4 lofi, 3 synthwave, 2 rock,
2 ambient, 2 jazz, 2 indie pop, and 1 folk** (20 songs total). A user
whose favorite genre is `folk` (only 1 song) will get essentially that
one song plus four "genre miss" fillers. **Users of under-represented
genres are systematically under-served.** Even the best-represented
genres (`pop` and `lofi`, 4 each) cover only 20% of the catalog.

### 5b. Small catalog

20 songs cannot represent any real genre. "Pop" here spans synth-pop,
dance-pop, and indie pop — three different sub-genres the model treats
as identical.

### 5c. Popularity bias

The `+0.5` popularity micro-bonus favors mainstream tracks. A user who
prefers obscure music gets no benefit from that term, and songs with
`popularity < 30` are effectively invisible if their other signals aren't
strong.

### 5d. Explicit preference model

The model requires the user to *know* their favorite genre and mood. Real
listeners often can't articulate this. A learned embedding would sidestep
the problem, at the cost of interpretability.

### 5e. Static catalog

There is no recency signal beyond a raw `release_year` column that the
scoring function doesn't use. A 2019 song and a 2024 song with the same
features score identically.

### 5f. OOP wrapper discards stretch attributes

The `Recommender` class (the OOP API used by `tests/test_recommender.py`)
converts each `Song` dataclass into a dict via `_to_dict()` and
**hardcodes** `popularity=50`, `release_year=2023`, `instrumentalness=0.0`,
and `liveness=0.0`. As a result:

- The popularity micro-bonus is **always exactly 0.25** in the OOP path.
- The 5 stretch attributes from the CSV are **ignored** by the OOP API.

The functional API (`score_song` / `recommend_songs`) does use the real
CSV values. This is a deliberate simplification to keep the dataclass
small, but it means the two APIs are **not score-equivalent**.

### 5g. Acoustic threshold is arbitrary

The binary fit uses `acousticness > 0.6` as the cutoff. This value is
hand-picked and not validated against user data. A song with
`acousticness = 0.59` is treated as "not acoustic" for a user who likes
acoustic music, despite being nearly identical to a song at `0.61`.

### 5h. No diversity penalty in the OOP path

The `Recommender.recommend()` method calls `recommend_songs()` with
default arguments, so the diversity penalty **is** applied. However, the
`Recommender` class does not expose a way to tune `diversity_weight` or
`mode` per call, unlike the functional API. Users of the OOP path are
locked into `genre-first` + `diversity_weight=0.8`.

---

## 6. Bias I Chose to Address (and Why)

**Filter bubble via repeated artists / genres.** The dataset has **8
artists** with multiple songs — 4 of them with 3 tracks each (Neon Echo,
LoRoom, Slow Stereo, Indigo Parade) and 4 more with 2 tracks each
(Voltline, Orbit Bloom, Paper Lanterns, Max Pulse). Without intervention,
a synthwave fan could see multiple consecutive tracks by the same artist
(e.g., *Night Drive Loop* and *After Hours* in the top-2).

**Fix:** a `diversity_weight=0.8` penalty is subtracted at picking time
for each previously seen artist and genre. Visible in `output.txt` as
`[diversity penalty: -0.80]` tags.

**Tradeoff:** the penalty is fixed at 0.8 regardless of how strong the
underlying signal is. A user who genuinely *wants* three tracks by the same
artist will be over-penalized. A per-user tunable would be the next step.

---

## 7. Ideas for Improvement

1. **Grow the catalog to 200+ songs** with balanced genre representation.
2. **Replace the fixed weights with learned weights** (e.g., logistic
   regression over past "liked" songs per user).
3. **Add collaborative filtering** so users with similar taste influence
   each other's recommendations.
4. **Add a novelty score** so the model can surface tracks the user is
   likely to *not* have heard yet.
5. **Make the diversity weight tunable** at the profile level, not global.
6. **Add release_year decay** so recent music gets a small bonus.
7. **Expand the OOP `Recommender` API** to carry all 15 CSV attributes
   (and accept `mode` / `diversity_weight`), so the two APIs are
   score-equivalent.
8. **Add a per-profile `max_tempo` test case** to exercise the tempo
   penalty in the CLI output (currently only covered by unit tests).

---

## 8. Coverage and AI Oversight Note

**When I asked ChatGPT and Gemini to help with the scoring logic, neither
model proactively flagged:**

- The genre imbalance in the dataset
- That `popularity` might favor mainstream bias
- That the diversity penalty has no per-user tunable
- That the OOP wrapper discards the stretch attributes it advertises
- That the `acousticness > 0.6` threshold is arbitrary and unvalidated

Both models implemented the code mechanics I asked for and stopped there.
**This is the same "silent coverage gap" pattern I documented in Module 2**
(see PawPal+ Model Card, section 7). AI is reliable as a *code assistant*
and unreliable as a *design reviewer* — it doesn't volunteer risks that
weren't explicitly requested.

**However**, when I later asked ChatGPT *"what inputs or user profiles
would make this score behave badly?"*, it produced a strong design
review that flagged **7 failure modes** — 4 of which I had not yet
documented (invalid numeric ranges, missing preferences creating
accidental matches, unbounded tempo penalties, and poorly calibrated
weights). This confirms that the value of AI as a design reviewer
depends entirely on the **prompt**: it will not volunteer risks, but it
will analyze them thoroughly when asked directly. See
`ai_interactions.md` (SF8) for the full exchange.

**My workflow change:** For this project, after every AI-suggested code
block I explicitly asked: *"What inputs or user profiles would make this
score behave badly?"*. That question surfaced the `folk` genre issue
before I shipped.

---

## 9. Rule-Based vs. ML Comparison (for context)

| Aspect | Rule-Based (this project) | Real ML Recommenders |
|---|---|---|
| **Interpretability** | High — every score is traceable | Low — embeddings are opaque |
| **Cold start** | Works immediately | Needs interaction data |
| **Data requirement** | None (rules written by hand) | Millions of interactions |
| **Failure mode** | Predictable ("unknown genre → 0") | Silent (high-confidence wrong) |
| **Cost** | Zero | GPU inference |

For a 20-song catalog with explicit user preferences, a rule-based system
is the correct design. A learned model would overfit and generalize worse.

---

## 10. Takeaway

**Ranking is not the same as judging.** A model that produces a plausible
top-5 is not a model that produces a fair one. The single most important
design decision in this project was **surfacing the reasons** next to each
score — that's what turned a black-box ranking into something a user can
audit and disagree with.