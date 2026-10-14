"""
Music Recommender Simulation — Full implementation.

Implements both the functional API (used by src/main.py) and the
OOP API (used by tests/test_recommender.py).

Base features:
  - load_songs()          -- CSV loader with type conversion
  - score_song()          -- weighted scoring with reasons
  - recommend_songs()     -- top-k sorted with explanations

Stretch features:
  - 3 ranking modes: "genre-first", "mood-first", "energy-similarity"
  - Diversity bonus: penalizes repeated artist / genre
  - Uses the 5 extra attributes from the expanded dataset
"""

import csv
from dataclasses import dataclass

# ---------------------------------------------------------------------
# Dataclasses (required by tests/test_recommender.py)
# ---------------------------------------------------------------------


@dataclass
class Song:
    """Represents a song and its attributes."""

    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""

    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


# ---------------------------------------------------------------------
# Ranking modes (stretch feature #3)
# ---------------------------------------------------------------------

RANKING_MODES = {
    "genre-first": {"genre": 3.0, "mood": 2.0, "energy": 2.0, "acoustic": 1.0},
    "mood-first": {"genre": 2.0, "mood": 3.0, "energy": 2.0, "acoustic": 1.0},
    "energy-similarity": {"genre": 1.0, "mood": 1.0, "energy": 4.0, "acoustic": 1.0},
}

DEFAULT_MODE = "genre-first"


# ---------------------------------------------------------------------
# Phase 1: load_songs — CSV loader with type conversion
# ---------------------------------------------------------------------


def load_songs(csv_path: str) -> list[dict]:
    """
    Load songs from a CSV file into a list of dicts.

    Converts numeric columns to their proper types.
    Keeps categorical columns (genre, mood, detailed_mood_tags) as strings.
    """
    songs: list[dict] = []
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                songs.append(
                    {
                        "id": int(row["id"]),
                        "title": row["title"],
                        "artist": row["artist"],
                        "genre": row["genre"],
                        "mood": row["mood"],
                        "energy": float(row["energy"]),
                        "tempo_bpm": float(row["tempo_bpm"]),
                        "valence": float(row["valence"]),
                        "danceability": float(row["danceability"]),
                        "acousticness": float(row["acousticness"]),
                        # Stretch attributes
                        "popularity": int(row["popularity"]),
                        "release_year": int(row["release_year"]),
                        "detailed_mood_tags": row["detailed_mood_tags"],
                        "instrumentalness": float(row["instrumentalness"]),
                        "liveness": float(row["liveness"]),
                    }
                )
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Songs file not found: {csv_path}") from err
    return songs


# ---------------------------------------------------------------------
# Phase 2/3: score_song — weighted scoring with reasons
# ---------------------------------------------------------------------


def score_song(
    user_prefs: dict,
    song: dict,
    mode: str = DEFAULT_MODE,
) -> tuple[float, list[str]]:
    """
    Score a single song against user preferences.

    Returns (score, reasons).
    """
    weights = RANKING_MODES.get(mode, RANKING_MODES[DEFAULT_MODE])
    score = 0.0
    reasons: list[str] = []

    # --- 1. Genre match ---
    if song["genre"] == user_prefs.get("favorite_genre"):
        score += weights["genre"]
        reasons.append(f"genre match ({song['genre']})")
    else:
        reasons.append(f"genre miss ({song['genre']})")

    # --- 2. Mood match ---
    if song["mood"] == user_prefs.get("favorite_mood"):
        score += weights["mood"]
        reasons.append(f"mood match ({song['mood']})")
    else:
        reasons.append(f"mood miss ({song['mood']})")

    # --- 3. Energy similarity (partial credit) ---
    target_energy = user_prefs.get("target_energy", 0.5)
    energy_diff = abs(song["energy"] - target_energy)
    energy_points = weights["energy"] * (1 - energy_diff)
    score += energy_points
    if energy_points > weights["energy"] * 0.7:
        reasons.append(f"great energy fit ({energy_points:.2f})")
    elif energy_points > weights["energy"] * 0.3:
        reasons.append(f"ok energy fit ({energy_points:.2f})")
    else:
        reasons.append(f"poor energy fit ({energy_points:.2f})")

    # --- 4. Acoustic preference ---
    likes_acoustic = user_prefs.get("likes_acoustic", False)
    is_acoustic = song["acousticness"] > 0.6
    if likes_acoustic == is_acoustic:
        score += weights["acoustic"]
        reasons.append("acoustic fits" if is_acoustic else "non-acoustic fits")
    else:
        reasons.append("acoustic preference mismatch")

    # --- 5. Popularity micro-bonus ---
    pop = song.get("popularity", 0)
    popularity_bonus = round(pop / 100 * 0.5, 3)
    score += popularity_bonus

    # --- 6. Optional tempo cap ---
    max_tempo = user_prefs.get("max_tempo")
    if max_tempo is not None and song["tempo_bpm"] > max_tempo:
        penalty = (song["tempo_bpm"] - max_tempo) / 100
        score -= penalty
        reasons.append(f"tempo over cap (-{penalty:.2f})")

    return round(score, 3), reasons


# ---------------------------------------------------------------------
# Phase 3/4: recommend_songs — top-k with diversity bonus (stretch #2)
# ---------------------------------------------------------------------


def recommend_songs(
    user_prefs: dict,
    songs: list[dict],
    k: int = 5,
    mode: str = DEFAULT_MODE,
    diversity_weight: float = 0.8,
) -> list[tuple[dict, float, str]]:
    """
    Return the top-k songs as (song_dict, score, explanation) tuples,
    sorted by ADJUSTED score descending.

    Diversity bonus: while selecting the top-k, penalize songs whose
    artist or genre already appeared in the picked set. This prevents
    the "same artist x5" filter bubble.

    The final list is re-sorted by the adjusted score so the returned
    list is always monotonically descending.
    """
    # Score every song first
    scored: list[tuple[dict, float, list[str]]] = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, mode=mode)
        scored.append((song, score, reasons))

    # Sort by raw score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Diversity-aware selection
    picked: list[tuple[dict, float, str]] = []
    seen_artists: dict[str, int] = {}
    seen_genres: dict[str, int] = {}

    for song, score, reasons in scored:
        if len(picked) >= k:
            break

        artist_repeat = seen_artists.get(song["artist"], 0)
        genre_repeat = seen_genres.get(song["genre"], 0)
        penalty = (artist_repeat + genre_repeat) * diversity_weight
        adjusted_score = round(score - penalty, 3)

        # Build a short explanation from the top reasons
        explanation = "; ".join(reasons[:3])

        # Only mention the diversity penalty if it actually applied
        if penalty > 0:
            explanation += f" [diversity penalty: -{penalty:.2f}]"

        picked.append((song, adjusted_score, explanation))
        seen_artists[song["artist"]] = artist_repeat + 1
        seen_genres[song["genre"]] = genre_repeat + 1

    # Re-sort by adjusted score so the returned list is always
    # monotonically descending.
    picked.sort(key=lambda x: x[1], reverse=True)

    return picked


# ---------------------------------------------------------------------
# OOP API (required by tests/test_recommender.py)
# ---------------------------------------------------------------------


class Recommender:
    """OOP implementation of the recommendation logic."""

    def __init__(self, songs: list[Song]):
        self.songs = songs

    def _to_dict(self, song: Song) -> dict:
        """Convert a Song dataclass to a dict for the functional scorer."""
        return {
            "id": song.id,
            "title": song.title,
            "artist": song.artist,
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "tempo_bpm": song.tempo_bpm,
            "valence": song.valence,
            "danceability": song.danceability,
            "acousticness": song.acousticness,
            "popularity": 50,
            "release_year": 2023,
            "detailed_mood_tags": "",
            "instrumentalness": 0.0,
            "liveness": 0.0,
        }

    def recommend(self, user: UserProfile, k: int = 5) -> list[Song]:
        """Return the top-k songs for a user (OOP API)."""
        prefs = {
            "favorite_genre": user.favorite_genre,
            "favorite_mood": user.favorite_mood,
            "target_energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        song_dicts = [self._to_dict(s) for s in self.songs]
        ranked = recommend_songs(prefs, song_dicts, k=k)

        by_id = {s.id: s for s in self.songs}
        return [by_id[item[0]["id"]] for item in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a short explanation of why a song was recommended."""
        prefs = {
            "favorite_genre": user.favorite_genre,
            "favorite_mood": user.favorite_mood,
            "target_energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        _, reasons = score_song(prefs, self._to_dict(song))
        return "; ".join(reasons)
