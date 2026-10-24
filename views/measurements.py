# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from db import delete_measurement, get_measurements, list_subjects, upsert_measurement
from utils.constants import STAGES, TEST_META, TEST_NAMES
from utils.ui import setup, subject_picker

setup("القياسات (T0–T3)", "🧪")
st.caption("إدخال نتائج الاختبارات في نقاط القياس الأربع. اترك الخانة فارغة إذا لم يُقَس المتغير.")

subjects = list_subjects()
if not subjects:
    st.warning("أضف لاعبين أولًا.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    subj = subject_picker(subjects)
with c2:
    stage = st.selectbox("نقطة القياس", STAGES)

existing = {m["test_key"]: m["value"] for m in get_measurements(subject_id=subj["id"], stage=stage)}

GROUPS_UI = {
    "الألم": ["nprs_move", "nprs_rest"],
    "مدى الحركة": [k for k in TEST_META if k.startswith("rom_")],
    "القوة": [k for k in TEST_META if k.startswith("str_")],
    "EMG/CCFT/JPE": ["emg_scm", "emg_scalene", "emg_trap", "ccft_level", "jpe"],
    "TSK-11": ["tsk11"],
    "PPT والتحمل": ["ppt_trap_r", "ppt_trap_l", "ppt_scm_r", "ppt_scm_l", "end_flex", "end_ext"],
    "التحكم الحركي": ["mc_track_rmse", "mc_sparc", "mc_coupled", "mc_vel_asym"],
    "المؤشرات المركّبة": ["nmci_balance", "nmci_motor", "nmci_total"],
}

with st.form("measure_form"):
    values = {}
    tabs = st.tabs(list(GROUPS_UI.keys()))
    for tab, (gname, keys) in zip(tabs, GROUPS_UI.items()):
        with tab:
            cols = st.columns(3)
            for i, k in enumerate(keys):
                name, unit, lo, hi, _ = TEST_META[k]
                values[k] = cols[i % 3].number_input(f"{name} [{unit}]", min_value=lo, max_value=hi,
                                                      value=existing.get(k), step=0.1, key=f"{k}_{subj['id']}_{stage}")
    notes = st.text_input("ملاحظات (اختياري)")
    if st.form_submit_button("💾 حفظ"):
        n = 0
        for k, v in values.items():
            if v is not None:
                upsert_measurement(subj["id"], stage, k, float(v), notes or None)
                n += 1
        st.success(f"حُفظ {n} قياسًا.")

st.divider()
st.subheader("سجل القياسات لهذا اللاعب")
recs = get_measurements(subject_id=subj["id"])
if recs:
    df = pd.DataFrame(recs)
    df["المتغير"] = df["test_key"].map(TEST_NAMES)
    piv = df.pivot_table(index="المتغير", columns="stage", values="value", aggfunc="first")
    st.dataframe(piv[[s for s in STAGES if s in piv.columns]], use_container_width=True)
    with st.expander("حذف قياس"):
        keys = sorted({(r["stage"], r["test_key"]) for r in recs})
        pick = st.selectbox("القياس", keys, format_func=lambda x: f"{x[0]} — {TEST_NAMES.get(x[1], x[1])}")
        if st.button("حذف"):
            delete_measurement(subj["id"], pick[0], pick[1])
            st.rerun()
else:
    st.info("لا توجد قياسات بعد.")
