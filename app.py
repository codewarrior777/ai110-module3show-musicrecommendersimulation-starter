"""
Streamlit UI for the Music Recommender Simulation.

Wraps the same functional API used by src/main.py and src/api.py.

Run locally:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src/ to sys.path so `from recommender import ...` works
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recommender import (
    RANKING_MODES,
    load_songs,
    recommend_songs,
)

# ---------------------------------------------------------------------
# Predefined profiles
# ---------------------------------------------------------------------
PROFILES = {
    "🎧 EDM / Gym Hero": {
        "favorite_genre": "pop",
        "favorite_mood": "intense",
        "target_energy": 0.9,
        "likes_acoustic": False,
    },
    "📚 Chill / Study Session": {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.35,
        "likes_acoustic": True,
    },
    "🌙 Late-Night Synthwave": {
        "favorite_genre": "synthwave",
        "favorite_mood": "moody",
        "target_energy": 0.7,
        "likes_acoustic": False,
    },
    "✏️ Custom profile": None,
}


# ---------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------
@st.cache_data
def get_catalog() -> list[dict]:
    """Load and cache the song catalog."""
    csv_path = Path(__file__).parent / "data" / "songs.csv"
    return load_songs(str(csv_path))


# ---------------------------------------------------------------------
# UI Components (Refactored into functions)
# ---------------------------------------------------------------------
def render_sidebar(songs: list[dict]) -> tuple[dict, str, float, int]:
    """Render the sidebar and return user choices."""
    st.sidebar.header("🎚️ Settings")

    profile_choice = st.sidebar.selectbox(
        "User profile",
        list(PROFILES.keys()),
        index=2,
    )

    if profile_choice == "✏️ Custom profile":
        st.sidebar.subheader("Custom preferences")
        custom_genre = st.sidebar.selectbox(
            "Favorite genre",
            sorted({s["genre"] for s in songs}),
        )
        custom_mood = st.sidebar.selectbox(
            "Favorite mood",
            sorted({s["mood"] for s in songs}),
        )
        custom_energy = st.sidebar.slider("Target energy", 0.0, 1.0, 0.7, 0.05)
        custom_acoustic = st.sidebar.checkbox("I like acoustic music", value=False)
        prefs = {
            "favorite_genre": custom_genre,
            "favorite_mood": custom_mood,
            "target_energy": custom_energy,
            "likes_acoustic": custom_acoustic,
        }
    else:
        prefs = PROFILES[profile_choice]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Ranking mode")

    mode = st.sidebar.radio(
        "How should songs be ranked?",
        list(RANKING_MODES.keys()),
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Diversity")

    diversity_weight = st.sidebar.slider(
        "Diversity penalty",
        0.0,
        3.0,
        0.8,
        0.1,
        help=(
            "How much to penalize repeated artists/genres. "
            "Set to 0.0 to disable (pure top-K by score)."
        ),
    )

    k = st.sidebar.slider("Number of recommendations", 1, 10, 5)

    return prefs, mode, diversity_weight, k


def render_profile_summary(prefs: dict):
    """Render the summary of the currently active profile."""
    st.subheader("📋 Profile in use")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Genre", prefs["favorite_genre"])
    col2.metric("Mood", prefs["favorite_mood"])
    col3.metric("Target energy", f"{prefs['target_energy']:.2f}")
    col4.metric("Acoustic?", "Yes" if prefs["likes_acoustic"] else "No")
    st.markdown("---")


def render_recommendations(
    songs: list[dict], prefs: dict, mode: str, diversity_weight: float, k: int
):
    """Render the recommendation table and expandable explanations."""
    st.subheader(f"🎯 Top {k} recommendations")
    st.caption(f"Ranking mode: `{mode}` · Diversity weight: `{diversity_weight}`")

    ranked = recommend_songs(
        user_prefs=prefs,
        songs=songs,
        k=k,
        mode=mode,
        diversity_weight=diversity_weight,
    )

    if not ranked:
        st.warning("No songs matched your profile.")
        return

    # Build the table
    table_rows = []
    for i, (song, score, _explanation) in enumerate(ranked, start=1):
        table_rows.append(
            {
                "#": i,
                "Title": song["title"],
                "Artist": song["artist"],
                "Genre": song["genre"],
                "Mood": song["mood"],
                "Energy": f"{song['energy']:.2f}",
                "Score": f"{score:.2f}",
            }
        )

    df = pd.DataFrame(table_rows)
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("💡 Why these songs?")

    # Build the expanders
    for i, (song, score, explanation) in enumerate(ranked, start=1):
        with st.expander(
            f"{i}. {song['title']} — {song['artist']} (Score: {score:.2f})"
        ):
            st.write(f"**Genre:** {song['genre']}")
            st.write(f"**Mood:** {song['mood']}")
            st.write(f"**Energy:** {song['energy']}")
            st.write(f"**Tempo:** {song['tempo_bpm']} BPM")
            st.write(f"**Acousticness:** {song['acousticness']}")
            st.write(f"**Popularity:** {song['popularity']}")
            st.write("---")
            st.write(f"**Why:** {explanation}")


# ---------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------
def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Music Recommender",
        page_icon="🎵",
        layout="wide",
    )

    st.title("🎵 Music Recommender Simulation")
    st.caption(
        "Scores and ranks a 20-song catalog against user taste profiles. "
        "Built for AI 110 — Foundations of AI Engineering."
    )
    st.markdown("---")

    # 1. Load Data
    songs = get_catalog()

    # 2. Render Sidebar
    prefs, mode, diversity_weight, k = render_sidebar(songs)

    # 3. Render Main Content
    render_profile_summary(prefs)
    render_recommendations(songs, prefs, mode, diversity_weight, k)

    # 4. Render Footer
    st.markdown("---")
    st.caption(
        "Built by Gustavo Ramos for AI 110 — Foundations of AI Engineering. "
        "[Source code](https://github.com/codewarrior777/"
        "ai110-module3show-musicrecommendersimulation-starter)"
    )


if __name__ == "__main__":
    main()