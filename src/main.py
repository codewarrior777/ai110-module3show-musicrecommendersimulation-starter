"""
Command line runner for the Music Recommender Simulation.

Runs 3 distinct user profiles through the recommender and prints the
top-5 songs per profile with explanations.

Uses `tabulate` for readable tables when available; falls back to plain
printing otherwise.
"""

import sys
from pathlib import Path

# Allow running from repo root or src/
sys.path.insert(0, str(Path(__file__).parent))

from recommender import load_songs, recommend_songs

PROFILES = {
    "EDM / Gym Hero": {
        "favorite_genre": "pop",
        "favorite_mood": "intense",
        "target_energy": 0.9,
        "likes_acoustic": False,
    },
    "Chill / Study Session": {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.35,
        "likes_acoustic": True,
    },
    "Late-Night Synthwave": {
        "favorite_genre": "synthwave",
        "favorite_mood": "moody",
        "target_energy": 0.7,
        "likes_acoustic": False,
    },
}


def print_plain(profile_name: str, ranked):
    print(f"\n=== {profile_name} ===\n")
    for i, (song, score, explanation) in enumerate(ranked, start=1):
        print(f"{i}. {song['title']} — {song['artist']} — Score: {score:.2f}")
        print(f"   Because: {explanation}\n")


def print_table(profile_name: str, ranked):
    try:
        from tabulate import tabulate
    except ImportError:
        return print_plain(profile_name, ranked)

    rows = [
        [i, song["title"], song["artist"], f"{score:.2f}", explanation[:60]]
        for i, (song, score, explanation) in enumerate(ranked, start=1)
    ]
    print(f"\n=== {profile_name} ===\n")
    print(
        tabulate(
            rows,
            headers=["#", "Title", "Artist", "Score", "Why (excerpt)"],
            tablefmt="fancy_grid",
        )
    )


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs.")

    for profile_name, prefs in PROFILES.items():
        for mode in ("genre-first", "mood-first", "energy-similarity"):
            ranked = recommend_songs(prefs, songs, k=5, mode=mode)
            print_table(f"{profile_name} — mode={mode}", ranked)


if __name__ == "__main__":
    main()
