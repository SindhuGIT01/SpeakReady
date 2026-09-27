"""SpeakReady Streamlit entry point (placeholder until Task 9)."""

import streamlit as st

from src import config


def main() -> None:
    """Render the placeholder home page and report config status."""
    st.set_page_config(page_title="SpeakReady", page_icon="🎤")
    st.title("🎤 SpeakReady")
    st.caption("AI voice interview coach — work in progress.")
    try:
        config.get_groq_api_key()
        st.success(f"Groq API key loaded. LLM model: {config.LLM_MODEL}")
    except config.ConfigError as err:
        st.error(str(err))


if __name__ == "__main__":
    main()
