"""
FastAPI REST API for the Music Recommender Simulation.

Exposes the recommender as REST endpoints with auto-generated Swagger
documentation at /docs.

Run with:
    uvicorn api:app --reload
Or:
    python api.py
"""

import sys
from pathlib import Path

# Ensure `src/` is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from recommender import (
    RANKING_MODES,
    load_songs,
    recommend_songs,
    score_song,
)

# ---------------------------------------------------------------------
# In-memory catalog (loaded once at startup)
# ---------------------------------------------------------------------

_CSV_PATH = Path(__file__).parent.parent / "data" / "songs.csv"
_SONGS = load_songs(str(_CSV_PATH)) if _CSV_PATH.exists() else []


# ---------------------------------------------------------------------
# Predefined user profiles (same as src/main.py)
# ---------------------------------------------------------------------

PROFILES = {
    "edm_gym": {
        "label": "EDM / Gym Hero",
        "prefs": {
            "favorite_genre": "pop",
            "favorite_mood": "intense",
            "target_energy": 0.9,
            "likes_acoustic": False,
        },
    },
    "chill_study": {
        "label": "Chill / Study Session",
        "prefs": {
            "favorite_genre": "lofi",
            "favorite_mood": "chill",
            "target_energy": 0.35,
            "likes_acoustic": True,
        },
    },
    "synthwave_late": {
        "label": "Late-Night Synthwave",
        "prefs": {
            "favorite_genre": "synthwave",
            "favorite_mood": "moody",
            "target_energy": 0.7,
            "likes_acoustic": False,
        },
    },
}


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

app = FastAPI(
    title="Music Recommender API",
    description=(
        "REST API for the Music Recommender Simulation. "
        "Score and rank a 20-song catalog against user taste profiles."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------


class UserPrefs(BaseModel):
    """User preferences for scoring songs."""

    favorite_genre: str = Field(..., json_schema_extra={"examples": ["pop"]})
    favorite_mood: str = Field(..., json_schema_extra={"examples": ["intense"]})
    target_energy: float = Field(
        ..., ge=0.0, le=1.0, json_schema_extra={"examples": [0.9]}
    )
    likes_acoustic: bool = Field(False, json_schema_extra={"examples": [False]})
    max_tempo: float | None = Field(None, json_schema_extra={"examples": [None]})


class RecommendationRequest(BaseModel):
    """Payload for POST /recommend."""

    user_prefs: UserPrefs
    k: int = Field(5, ge=1, le=20)
    mode: str = Field("genre-first", json_schema_extra={"examples": ["genre-first"]})
    diversity_weight: float = Field(0.8, ge=0.0, le=5.0)


class SongOut(BaseModel):
    """A song in API responses."""

    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _song_to_out(song: dict) -> SongOut:
    return SongOut(
        id=song["id"],
        title=song["title"],
        artist=song["artist"],
        genre=song["genre"],
        mood=song["mood"],
        energy=song["energy"],
        tempo_bpm=song["tempo_bpm"],
    )


def _check_mode(mode: str) -> None:
    if mode not in RANKING_MODES:
        raise HTTPException(
            400,
            f"Unknown mode '{mode}'. Choose from: {list(RANKING_MODES.keys())}",
        )


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------


@app.get("/", tags=["Health"])
def root():
    """Health check + link to interactive docs."""
    return {
        "app": "Music Recommender API",
        "status": "ok",
        "version": "1.0.0",
        "song_count": len(_SONGS),
        "docs": "/docs",
    }


# ---------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------


@app.get("/songs", tags=["Catalog"])
def list_songs():
    """List all songs in the catalog."""
    return [_song_to_out(s) for s in _SONGS]


@app.get("/songs/{song_id}", tags=["Catalog"])
def get_song(song_id: int):
    """Return a single song by ID."""
    for s in _SONGS:
        if s["id"] == song_id:
            return _song_to_out(s)
    raise HTTPException(404, f"Song id={song_id} not found")


# ---------------------------------------------------------------------
# Modes and profiles
# ---------------------------------------------------------------------


@app.get("/modes", tags=["Metadata"])
def list_modes():
    """List available ranking modes with their weights."""
    return RANKING_MODES


@app.get("/profiles", tags=["Metadata"])
def list_profiles():
    """List the three built-in user profiles."""
    return {
        key: {"label": p["label"], "prefs": p["prefs"]} for key, p in PROFILES.items()
    }


# ---------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------


@app.post("/recommend", tags=["Recommendation"])
def recommend(payload: RecommendationRequest):
    """
    Recommend the top-K songs for a custom user profile.

    Returns (song, score, explanation) tuples sorted by score descending.
    """
    _check_mode(payload.mode)
    prefs = payload.user_prefs.model_dump(exclude_none=True)
    ranked = recommend_songs(
        prefs,
        _SONGS,
        k=payload.k,
        mode=payload.mode,
        diversity_weight=payload.diversity_weight,
    )
    return [
        {
            "song": _song_to_out(song).model_dump(),
            "score": score,
            "explanation": explanation,
        }
        for song, score, explanation in ranked
    ]


@app.get("/recommend/{profile_key}", tags=["Recommendation"])
def recommend_for_profile(
    profile_key: str,
    k: int = Query(5, ge=1, le=20, description="Number of songs to return (1-20)"),
    mode: str = Query("genre-first", description="Ranking mode"),
):
    """
    Recommend the top-K songs for one of the built-in profiles.

    Valid profile keys: edm_gym, chill_study, synthwave_late.
    """
    if profile_key not in PROFILES:
        raise HTTPException(
            404,
            f"Unknown profile '{profile_key}'. Choose from: {list(PROFILES.keys())}",
        )
    _check_mode(mode)
    prefs = PROFILES[profile_key]["prefs"]
    ranked = recommend_songs(prefs, _SONGS, k=k, mode=mode)
    return {
        "profile": PROFILES[profile_key]["label"],
        "mode": mode,
        "results": [
            {
                "song": _song_to_out(song).model_dump(),
                "score": score,
                "explanation": explanation,
            }
            for song, score, explanation in ranked
        ],
    }


# ---------------------------------------------------------------------
# Score a single song
# ---------------------------------------------------------------------


@app.post("/score/{song_id}", tags=["Recommendation"])
def score_one(song_id: int, payload: UserPrefs):
    """Score a single song against a user profile."""
    for s in _SONGS:
        if s["id"] == song_id:
            score, reasons = score_song(payload.model_dump(exclude_none=True), s)
            return {
                "song": _song_to_out(s).model_dump(),
                "score": score,
                "reasons": reasons,
            }
    raise HTTPException(404, f"Song id={song_id} not found")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
