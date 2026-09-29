"""SpeakReady Streamlit entry point: page config and navigation."""

from __future__ import annotations

import streamlit as st

from app_common import ensure_question_bank_ingested, init_session_state

st.set_page_config(
    page_title="SpeakReady",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
ensure_question_bank_ingested()

home_page = st.Page("app_pages/home.py", title="Home", icon="🏠", default=True)
setup_page = st.Page("app_pages/setup.py", title="Setup", icon="⚙️")
interview_page = st.Page("app_pages/interview.py", title="Interview", icon="🎙️")
report_page = st.Page("app_pages/report.py", title="Report", icon="📊")
progress_page = st.Page("app_pages/progress.py", title="Progress", icon="📈")

with st.sidebar:
    st.markdown("### 🎤 SpeakReady")
    st.caption("Your AI voice interview coach")
    st.divider()

navigation = st.navigation([home_page, setup_page, interview_page, report_page, progress_page])
navigation.run()
