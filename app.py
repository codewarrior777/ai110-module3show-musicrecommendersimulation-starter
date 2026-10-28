"""
Streamlit UI for the Music Recommender Simulation.

Wraps the same functional API used by src/main.py and src/api.py.

Run locally:
    streamlit run app.py
"""

import sys
from pathlib import Path

import google.generativeai as genai
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
# AI Assistant Configuration
# ---------------------------------------------------------------------
try:
    # Lee la clave desde el archivo .streamlit/secrets.toml
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Usamos Gemini 1.5 Flash por su rapidez y eficiencia
    ai_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    ai_model = None
    ai_error = str(e)


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
# UI Components
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


def render_chat_assistant():
    """Render the AI chat interface."""
    st.header("🤖 Music Recommender Assistant")
    st.caption(
        "Ask me about the catalog, the ranking modes, or how the algorithm works."
    )

    if not ai_model:
        st.error(f"AI Assistant is not configured. Error: {ai_error}")
        st.info(
            "Please make sure you have added your GEMINI_API_KEY to `.streamlit/secrets.toml`."
        )
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to know?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                system_prompt = (
                    "You are an expert assistant for the 'Music Recommender Simulation' project. "
                    "Your knowledge is based on the 20 songs in `data/songs.csv` and the scoring logic in `src/recommender.py`. "
                    "You can explain the three ranking modes: 'genre-first', 'mood-first', and 'energy-similarity'. "
                    "You can also explain how the 'diversity penalty' works to prevent filter bubbles. "
                    "Keep your answers concise, helpful, and directly related to this specific project."
                )

                # Combine context and prompt
                full_prompt = f"{system_prompt}\n\nUser Question: {prompt}"

                response = ai_model.generate_content(full_prompt)

                message_placeholder.markdown(response.text)
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.text}
                )
            except Exception as e:
                message_placeholder.error(f"Sorry, I encountered an error: {e}")


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

    # Create the two tabs
    tab1, tab2 = st.tabs(["🎵 Recommender", "🤖 AI Assistant"])

    with tab1:
        # 1. Load Data
        songs = get_catalog()

        # 2. Render Sidebar
        prefs, mode, diversity_weight, k = render_sidebar(songs)

        # 3. Render Main Content
        render_profile_summary(prefs)
        render_recommendations(songs, prefs, mode, diversity_weight, k)

    with tab2:
        # 4. Render AI Chat
        render_chat_assistant()

    # 5. Render Footer
    st.markdown("---")
    st.caption(
        "Built by Gustavo Ramos for AI 110 — Foundations of AI Engineering. "
        "[Source code](https://github.com/codewarrior777/"
        "ai110-module3show-musicrecommendersimulation-starter)"
    )


if __name__ == "__main__":
    main()
