"""Benchmark script for the Music Recommender.

Measures how `score_song` and `recommend_songs` scale with catalog size.

Run:
    python benchmark.py
"""

import sys
import time
from pathlib import Path

from tabulate import tabulate

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recommender import load_songs, recommend_songs, score_song


PROFILE = {
    "favorite_genre": "synthwave",
    "favorite_mood": "moody",
    "target_energy": 0.7,
    "likes_acoustic": False,
}


def _time_it(func, *args, runs: int = 5, **kwargs) -> float:
    """Run func N times, return the best (fastest) duration in ms."""
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        best = min(best, elapsed)
    return round(best, 4)


def _expand_catalog(base_songs: list, n: int) -> list:
    """Duplicate the base catalog until it has at least n songs."""
    expanded = []
    i = 0
    while len(expanded) < n:
        song = dict(base_songs[i % len(base_songs)])
        song["id"] = len(expanded) + 1
        expanded.append(song)
        i += 1
    return expanded


def main() -> None:
    base_songs = load_songs("data/songs.csv")
    sizes = [20, 100, 500, 1000, 2000]

    print("\n🎵 Music Recommender — Benchmark")
    print("=" * 80)
    print("Times shown in milliseconds (best of 5 runs)\n")

    headers = ["Operation"] + [f"n={n}" for n in sizes]
    rows = []

    score_row = ["score_song (single)"]
    for n in sizes:
        songs = _expand_catalog(base_songs, n)
        song = songs[0]
        ms = _time_it(score_song, PROFILE, song)
        score_row.append(f"{ms:.4f}")
    rows.append(score_row)

    rec_row = ["recommend_songs (top-5)"]
    for n in sizes:
        songs = _expand_catalog(base_songs, n)
        ms = _time_it(recommend_songs, PROFILE, songs, k=5)
        rec_row.append(f"{ms:.2f}")
    rows.append(rec_row)

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    print("\nExpected scaling:")
    print("  • score_song      → O(1) per song")
    print("  • recommend_songs → O(n log n) (dominant cost: sorting)")
    print("\nNote: n>20 uses duplicated songs to test scaling.")


if __name__ == "__main__":
    main()