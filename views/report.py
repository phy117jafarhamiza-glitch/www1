# -*- coding: utf-8 -*-
import streamlit as st

from db import get_measurements, list_ai_sets, list_coach_sessions, list_emg_analyses, list_pain, list_sessions, list_subjects
from utils.report import generate_subject_report
from utils.ui import setup, subject_picker

setup("تقرير اللاعب", "📄")
subjects = list_subjects()
if not subjects:
    st.warning("أضف لاعبين أولًا.")
    st.stop()
subj = subject_picker(subjects)
if st.button("إنشاء التقرير"):
    path = generate_subject_report(subj, get_measurements(subject_id=subj["id"]),
                                   list_emg_analyses(subj["id"]), list_ai_sets(subj["id"]),
                                   sessions=list_sessions(subj["id"]), coach=list_coach_sessions(subj["id"]),
                                   pain=list_pain(subj["id"]))
    with open(path, "rb") as f:
        st.session_state["report_bytes"] = f.read()
if st.session_state.get("report_bytes"):
    st.download_button("⬇️ تنزيل التقرير (Word)", st.session_state["report_bytes"], f"report_{subj['id']}.docx")
