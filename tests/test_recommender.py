"""
Test suite for the Music Recommender Simulation.

Covers:
  - load_songs (CSV loading, type conversion, FileNotFoundError)
  - score_song (all scoring branches: genre, mood, energy, acoustic,
    popularity, tempo)
  - recommend_songs (sorting, top-k, ranking modes, diversity penalty)
  - Recommender OOP wrapper (recommend, explain_recommendation)
"""

import pytest

from src.recommender import (
    RANKING_MODES,
    Recommender,
    Song,
    UserProfile,
    load_songs,
    recommend_songs,
    score_song,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_song() -> dict:
    """A fully-populated song dict (matches load_songs output shape)."""
    return {
        "id": 1,
        "title": "Test Pop Track",
        "artist": "Test Artist",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "tempo_bpm": 120.0,
        "valence": 0.9,
        "danceability": 0.8,
        "acousticness": 0.2,
        "popularity": 70,
        "release_year": 2023,
        "detailed_mood_tags": "upbeat,euphoric",
        "instrumentalness": 0.05,
        "liveness": 0.15,
    }


@pytest.fixture
def sample_profile() -> dict:
    """A user preferences dict."""
    return {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
    }


@pytest.fixture
def small_recommender() -> Recommender:
    """A Recommender with 2 songs for OOP tests."""
    songs = [
        Song(
            id=1, title="Test Pop Track", artist="Test Artist",
            genre="pop", mood="happy", energy=0.8, tempo_bpm=120,
            valence=0.9, danceability=0.8, acousticness=0.2,
        ),
        Song(
            id=2, title="Chill Lofi Loop", artist="Other Artist",
            genre="lofi", mood="chill", energy=0.4, tempo_bpm=80,
            valence=0.6, danceability=0.5, acousticness=0.9,
        ),
    ]
    return Recommender(songs)


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

class TestDataclasses:
    def test_song_creation(self):
        s = Song(
            id=1, title="X", artist="Y", genre="pop", mood="happy",
            energy=0.8, tempo_bpm=120, valence=0.9,
            danceability=0.8, acousticness=0.2,
        )
        assert s.id == 1
        assert s.title == "X"
        assert s.genre == "pop"

    def test_user_profile_creation(self):
        u = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        assert u.favorite_genre == "pop"
        assert u.likes_acoustic is False


# ---------------------------------------------------------------------
# load_songs
# ---------------------------------------------------------------------

class TestLoadSongs:
    def test_loads_real_csv(self):
        songs = load_songs("data/songs.csv")
        assert len(songs) == 20
        assert songs[0]["title"] == "Sunrise City"
        assert isinstance(songs[0]["energy"], float)
        assert isinstance(songs[0]["id"], int)

    def test_loads_all_15_attributes(self):
        songs = load_songs("data/songs.csv")
        first = songs[0]
        expected = {
            "id", "title", "artist", "genre", "mood",
            "energy", "tempo_bpm", "valence", "danceability", "acousticness",
            "popularity", "release_year", "detailed_mood_tags",
            "instrumentalness", "liveness",
        }
        assert set(first.keys()) == expected

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Songs file not found"):
            load_songs("data/does_not_exist.csv")


# ---------------------------------------------------------------------
# score_song — branch coverage
# ---------------------------------------------------------------------

class TestScoreSong:
    def test_all_signals_match(self, sample_profile, sample_song):
        score, reasons = score_song(sample_profile, sample_song)
        assert score > 0
        assert any("genre match" in r for r in reasons)
        assert any("mood match" in r for r in reasons)
        assert any("great energy fit" in r for r in reasons)
        assert any("non-acoustic fits" in r for r in reasons)

    def test_all_signals_miss(self, sample_song):
        prefs = {
            "favorite_genre": "jazz",
            "favorite_mood": "angry",
            "target_energy": 0.0,
            "likes_acoustic": True,
        }
        score, reasons = score_song(prefs, sample_song)
        assert any("genre miss" in r for r in reasons)
        assert any("mood miss" in r for r in reasons)
        assert any("poor energy fit" in r for r in reasons)
        assert any("mismatch" in r for r in reasons)

    def test_ok_energy_fit_branch(self, sample_song):
        # target 0.3, song 0.8 → diff 0.5 → falls in "ok" band
        prefs = {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.3,
            "likes_acoustic": False,
        }
        _, reasons = score_song(prefs, sample_song)
        assert any("ok energy fit" in r for r in reasons)

    def test_acoustic_preference_match(self, sample_song):
        # song acousticness = 0.2 (<0.6), likes_acoustic=True → mismatch
        prefs = {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.8,
            "likes_acoustic": True,
        }
        _, reasons = score_song(prefs, sample_song)
        assert any("mismatch" in r for r in reasons)

    def test_popularity_bonus_is_applied(self, sample_song):
        low = {**sample_song, "popularity": 0}
        high = {**sample_song, "popularity": 100}
        prefs = {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.8,
            "likes_acoustic": False,
        }
        score_low, _ = score_song(prefs, low)
        score_high, _ = score_song(prefs, high)
        assert score_high > score_low

    def test_tempo_cap_penalty(self, sample_profile, sample_song):
        prefs = {**sample_profile, "max_tempo": 100.0}
        # song tempo is 120 → over cap by 20 → penalty = 0.2
        _, reasons = score_song(prefs, sample_song)
        assert any("tempo over cap" in r for r in reasons)

    def test_tempo_under_cap_no_penalty(self, sample_profile, sample_song):
        prefs = {**sample_profile, "max_tempo": 200.0}
        _, reasons = score_song(prefs, sample_song)
        assert not any("tempo over cap" in r for r in reasons)

    def test_each_mode_returns_different_score(self, sample_profile, sample_song):
        s_genre, _ = score_song(sample_profile, sample_song, mode="genre-first")
        s_energy, _ = score_song(
            sample_profile, sample_song, mode="energy-similarity"
        )
        assert s_genre != s_energy

    def test_unknown_mode_falls_back_to_default(self, sample_profile, sample_song):
        s_default, _ = score_song(sample_profile, sample_song)
        s_unknown, _ = score_song(
            sample_profile, sample_song, mode="does-not-exist"
        )
        assert s_default == s_unknown


# ---------------------------------------------------------------------
# recommend_songs
# ---------------------------------------------------------------------

class TestRecommendSongs:
    def test_returns_top_k(self, sample_profile):
        songs = load_songs("data/songs.csv")
        results = recommend_songs(sample_profile, songs, k=3)
        assert len(results) == 3

    def test_sorted_descending_by_score(self, sample_profile):
        songs = load_songs("data/songs.csv")
        results = recommend_songs(sample_profile, songs, k=5)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_each_result_is_triple(self, sample_profile):
        songs = load_songs("data/songs.csv")
        results = recommend_songs(sample_profile, songs, k=2)
        for song, score, explanation in results:
            assert isinstance(song, dict)
            assert isinstance(score, (int, float))
            assert isinstance(explanation, str)
            assert explanation != ""

    def test_diversity_penalty_visible(self):
        # Synthwave profile → multiple synthwave songs by same artists
        prefs = {
            "favorite_genre": "synthwave",
            "favorite_mood": "moody",
            "target_energy": 0.7,
            "likes_acoustic": False,
        }
        songs = load_songs("data/songs.csv")
        results = recommend_songs(prefs, songs, k=5)
        # At least one should have the diversity penalty tag
        assert any("diversity penalty" in r[2] for r in results)

    def test_diversity_weight_zero_disables_penalty(self):
        prefs = {
            "favorite_genre": "synthwave",
            "favorite_mood": "moody",
            "target_energy": 0.7,
            "likes_acoustic": False,
        }
        songs = load_songs("data/songs.csv")
        results = recommend_songs(prefs, songs, k=5, diversity_weight=0.0)
        assert not any("diversity penalty" in r[2] for r in results)

    def test_different_modes_produce_different_orderings(self):
        prefs = {
            "favorite_genre": "synthwave",
            "favorite_mood": "moody",
            "target_energy": 0.7,
            "likes_acoustic": False,
        }
        songs = load_songs("data/songs.csv")
        r_genre = recommend_songs(prefs, songs, k=5, mode="genre-first")
        r_energy = recommend_songs(
            prefs, songs, k=5, mode="energy-similarity"
        )
        assert [r[0]["id"] for r in r_genre] != [r[0]["id"] for r in r_energy]


# ---------------------------------------------------------------------
# RANKING_MODES metadata
# ---------------------------------------------------------------------

class TestRankingModes:
    def test_all_modes_defined(self):
        assert "genre-first" in RANKING_MODES
        assert "mood-first" in RANKING_MODES
        assert "energy-similarity" in RANKING_MODES

    def test_energy_similarity_prioritizes_energy(self):
        assert (
            RANKING_MODES["energy-similarity"]["energy"]
            > RANKING_MODES["energy-similarity"]["genre"]
        )

    def test_genre_first_prioritizes_genre(self):
        assert (
            RANKING_MODES["genre-first"]["genre"]
            > RANKING_MODES["genre-first"]["mood"]
        )


# ---------------------------------------------------------------------
# Recommender OOP wrapper
# ---------------------------------------------------------------------

class TestRecommenderOOP:
    def test_recommend_returns_top_k(self, small_recommender):
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        results = small_recommender.recommend(user, k=2)
        assert len(results) == 2
        assert isinstance(results[0], Song)

    def test_recommend_prefers_matching_genre(self, small_recommender):
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        results = small_recommender.recommend(user, k=2)
        assert results[0].genre == "pop"
        assert results[0].mood == "happy"

    def test_explain_recommendation_returns_string(self, small_recommender):
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        song = small_recommender.songs[0]
        explanation = small_recommender.explain_recommendation(user, song)
        assert isinstance(explanation, str)
        assert explanation.strip() != ""

    def test_explain_recommendation_mentions_genre(self, small_recommender):
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        song = small_recommender.songs[0]
        explanation = small_recommender.explain_recommendation(user, song)
        assert "pop" in explanation

    def test_recommend_returns_objects_with_all_fields(self, small_recommender):
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        results = small_recommender.recommend(user, k=1)
        song = results[0]
        assert hasattr(song, "title")
        assert hasattr(song, "artist")
        assert hasattr(song, "genre")
        assert hasattr(song, "mood")