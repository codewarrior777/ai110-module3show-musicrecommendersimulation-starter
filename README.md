# 🎵 Music Recommender Simulation

A command-line music recommender that scores and ranks a 20-song catalog
against user taste profiles. Built as Project 3 for **AI 110 — Foundations
of AI Engineering**.

---

## 1. How Music Recommendation Systems Work

Real-world recommenders like **Spotify, YouTube Music, and Apple Music**
do not simply match "genre" to "genre". They combine several layers:

### 1a. Data features

Streaming services collect two broad categories of features for every track:

- **Audio features** — extracted by analyzing the audio waveform itself
  (energy, tempo, danceability, acousticness, valence, instrumentalness).
  These are computed by ML models trained to predict perceptual qualities.
- **Context features** — metadata about the track and user (genre,
  release year, popularity, listening history, skip rate, save rate).

### 1b. User preferences

A modern recommender learns a **user taste vector** from past behavior —
which songs were played fully, which were skipped, which were added to
playlists. It does not require the user to fill out a form. Over time, the
model approximates the user's centroid in feature space.

### 1c. Ranking and selection

Given a user vector, the system scores every candidate track and returns
the **top-K** by score. Real systems add constraints on top of the score:

- **Diversity** (avoid recommending five songs by the same artist)
- **Novelty** (surface tracks the user hasn't heard yet)
- **Freshness** (bias toward recently released music)

### 1d. How this project differs from the real thing

This project is a **simplified, explicit-feature version** of the pipeline
above. Instead of learning embeddings from listening data, we:

1. Store features explicitly in a CSV (20 songs, 15 attributes each).
2. Ask the user for their preferences explicitly (genre, mood, energy).
3. Use a **weighted scoring function** instead of a learned model.
4. Apply a **diversity penalty** at ranking time to prevent filter bubbles.

The simplification matters: it lets us inspect **exactly why** a song was
recommended, which is impossible inside a black-box neural recommender.

---

## 2. The Algorithm Recipe (Phase 2)

### 2a. Feature schema

Every song in `data/songs.csv` has these attributes:

| Attribute | Type | Range | Description |
|---|---|---|---|
| `id` | int | 1-20 | Unique identifier |
| `title` | str | — | Song title |
| `artist` | str | — | Artist name |
| `genre` | str | — | `pop`, `lofi`, `rock`, `ambient`, `jazz`, `synthwave`, `indie pop`, `folk` |
| `mood` | str | — | `happy`, `chill`, `intense`, `relaxed`, `focused`, `moody`, `angry` |
| `energy` | float | 0.0-1.0 | Perceptual intensity |
| `tempo_bpm` | float | 60-160 | Beats per minute |
| `valence` | float | 0.0-1.0 | Musical positivity |
| `danceability` | float | 0.0-1.0 | How suitable for dancing |
| `acousticness` | float | 0.0-1.0 | Confidence the track is acoustic |
| `popularity` | int | 0-100 | Stream-based popularity score |
| `release_year` | int | 2019-2024 | Year released |
| `detailed_mood_tags` | str | — | Comma-separated mood keywords |
| `instrumentalness` | float | 0.0-1.0 | Likelihood of no vocals |
| `liveness` | float | 0.0-1.0 | Presence of audience |

### 2b. Scoring formula

For each song, `score_song(user_prefs, song)` computes:

```
score =  (weights["genre"]    if genre == user.favorite_genre   else 0)
       + (weights["mood"]     if mood  == user.favorite_mood    else 0)
       +  weights["energy"] * (1 - abs(song.energy - user.target_energy))
       + (weights["acoustic"] if acoustic_pref_matches          else 0)
       +  song.popularity / 100 * 0.5        # popularity micro-bonus
       -  tempo_penalty (if user.max_tempo set)
```

**Weights vary by mode:**

| Mode | genre | mood | energy | acoustic |
|---|---|---|---|---|
| `genre-first` (default) | **3.0** | 2.0 | 2.0 | 1.0 |
| `mood-first` | 2.0 | **3.0** | 2.0 | 1.0 |
| `energy-similarity` | 1.0 | 1.0 | **4.0** | 1.0 |

### 2c. Diversity bonus (stretch feature)

The recommendation function re-ranks candidates with a diversity penalty:

```
adjusted_score = raw_score - (artist_repeats + genre_repeats) * 0.8
```

Every time the same artist or genre appears again in the picked set, the
penalty grows. This prevents the **"five songs by the same artist"** filter
bubble that a pure top-K selection would produce.

---

## 3. Evaluation — 3 Profiles × 3 Modes

The recommender was run with 3 distinct user profiles across all 3
ranking modes. Full output is in `output.txt`.

### 3a. User profiles

| Profile | Genre | Mood | Target Energy | Acoustic |
|---|---|---|---|---|
| **EDM / Gym Hero** | pop | intense | 0.90 | No |
| **Chill / Study Session** | lofi | chill | 0.35 | Yes |
| **Late-Night Synthwave** | synthwave | moody | 0.70 | No |

### 3b. Sample output — Late-Night Synthwave, mode=`genre-first`

```
=== Late-Night Synthwave — mode=genre-first ===

1. After Hours — LoRoom — Score: 8.27
   Because: genre match (synthwave); mood match (moody); great energy fit (1.96)

2. Night Drive Loop — Neon Echo — Score: 7.45
   Because: genre match (synthwave); mood match (moody); great energy fit (1.90) [diversity penalty: -0.80]

3. Neon Alley — Slow Stereo — Score: 4.47
   Because: genre match (synthwave); mood miss (chill); great energy fit (1.76) [diversity penalty: -1.60]

4. Paper Planes — Indigo Parade — Score: 5.00
   Because: genre miss (indie pop); mood match (moody); great energy fit (1.70)

5. Midnight Jazz — Slow Stereo — Score: 3.09
   Because: genre miss (jazz); mood match (moody); great energy fit (1.60) [diversity penalty: -0.80]
```

### 3c. Behavior differences across profiles

- **EDM / Gym Hero** favors high-energy pop (Gym Hero, Rooftop Reprise).
  Acoustic songs appear at the bottom with a mismatch penalty.
- **Chill / Study Session** favors lofi with high acousticness
  (Library Rain, Focus Flow, Study Hall). Danceable pop ranks low.
- **Late-Night Synthwave** favors moody synthwave (After Hours, Night
  Drive Loop). The diversity penalty kicks in on the 2nd track and shows
  **visible score reductions** when the same artist appears again.

### 3d. Behavior differences across modes

- **`genre-first`** — genre is the strongest signal (weight 3.0). Songs
  with the wrong genre rarely crack the top-3, even if mood and energy fit.
- **`mood-first`** — mood takes over as the primary key. A song with the
  right mood but wrong genre (e.g., *Paper Planes* in the Synthwave profile)
  jumps up the ranking.
- **`energy-similarity`** — genre and mood matter less; the closest
  energy match wins. This surfaces songs a genre-first ranking would miss.

---

## 4. How to Run

```bash
# Clone
git clone https://github.com/codewarrior777/ai110-module3show-musicrecommendersimulation-starter.git
cd ai110-module3show-musicrecommendersimulation-starter

# Set up virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run the recommender
python src/main.py
```

---

## 5. Project Structure

```
ai110-module3show-musicrecommendersimulation-starter/
├── data/
│   └── songs.csv          # 20 songs, 15 attributes each
├── src/
│   ├── recommender.py     # Core logic: load, score, rank
│   └── main.py            # CLI runner (3 profiles × 3 modes)
├── tests/                 # Placeholder for future tests
├── output.txt             # Saved CLI output for all combinations
├── model_card.md          # Model Card (bias, limits, improvement)
├── ai_interactions.md     # AI agent workflow log
├── requirements.txt
└── README.md
```

---

## 🙏 Credits

Built by **Gustavo Ramos** for *AI 110 — Foundations of AI Engineering*.

*"The bugs were the curriculum."*