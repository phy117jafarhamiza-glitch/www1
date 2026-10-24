# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from db import add_pain, list_pain, list_subjects, upsert_measurement
from utils.constants import EXERCISE_BY_ID, PRIMARY_KEY, STAGES
from utils.kinematics import symmetry_index
from utils.ui import setup, subject_picker

setup("تسجيل الألم اللحظي وعتبة ألم الضغط", "🩺")
st.caption("NPRS لحظي أثناء التمرين مع ربطه بالزاوية والـ EMG، وقياس عتبة ألم الضغط (PPT) بمقياس algometer.")
subjects = list_subjects()
if not subjects:
    st.warning("أضف لاعبين أولًا.")
    st.stop()
subj = subject_picker(subjects)
tab1, tab2, tab3 = st.tabs(["الألم اللحظي", "عتبة ألم الضغط (PPT)", "علاقة الألم بالحركة والـ EMG"])

with tab1:
    c = st.columns(4)
    week = c[0].number_input("الأسبوع", 1, 8, 1)
    sess = c[1].number_input("الجلسة", 1, 3, 1)
    exid = c[2].selectbox("التمرين", list(EXERCISE_BY_ID), format_func=lambda i: f"{i}. {EXERCISE_BY_ID[i]['name']}")
    setn = c[3].number_input("المجموعة", 1, 6, 1)
    nprs = st.slider("شدة الألم الآن (NPRS 0–10)", 0, 10, 0)
    c = st.columns(3)
    ang = c[0].number_input("الزاوية الحالية (°) — اختياري", 0.0, 150.0, None)
    emg = c[1].number_input("EMG % MVC — اختياري", 0.0, 300.0, None)
    note = c[2].text_input("ملاحظة")
    rows = [r for r in list_pain(subj["id"]) if r["week"] == int(week) and r["session_no"] == int(sess)]
    start = rows[0]["nprs"] if rows else nprs
    if nprs >= 6 or (nprs - start) >= 2:
        st.error("⛔ الألم ≥ 6/10 أو ارتفع نقطتين عن بداية الجلسة: أوقف المجموعة وبلّغ الأخصائي.")
    if st.button("➕ تسجيل"):
        add_pain(subj["id"], int(week), int(sess), int(exid), int(setn), float(nprs), ang, emg, note or None)
        st.rerun()
    if rows:
        df = pd.DataFrame(rows)
        st.line_chart(df.set_index(df.index + 1)["nprs"])
        st.dataframe(df.drop(columns=["subject_id"]), use_container_width=True)
        st.metric("أعلى ألم في الجلسة", f"{df['nprs'].max():.0f}")

with tab2:
    st.markdown("ثلاث محاولات لكل موضع؛ يُحفظ المتوسط كمتغير في نقطة القياس المختارة.")
    stage = st.selectbox("نقطة القياس", STAGES, key="ppt_stage")
    sites = {"ppt_trap_r": "شبه المنحرفة — يمين", "ppt_trap_l": "شبه المنحرفة — يسار",
             "ppt_scm_r": "القصية الترقوية — يمين", "ppt_scm_l": "القصية الترقوية — يسار"}
    means = {}
    cols = st.columns(4)
    for col, (k, lab) in zip(cols, sites.items()):
        col.markdown(f"**{lab}**")
        t = [col.number_input(f"محاولة {i + 1}", 0.0, 1000.0, None, key=f"{k}{i}") for i in range(3)]
        t = [x for x in t if x is not None]
        if t:
            means[k] = float(np.mean(t))
            col.caption(f"المتوسط: {means[k]:.0f} kPa")
    for a, b, lab in (("ppt_trap_r", "ppt_trap_l", "شبه المنحرفة"), ("ppt_scm_r", "ppt_scm_l", "القصية الترقوية")):
        if a in means and b in means:
            st.write(f"عدم تماثل {lab}: **{symmetry_index(means[a], means[b]):.1f}%**")
    if means and st.button("💾 حفظ PPT"):
        for k, v in means.items():
            upsert_measurement(subj["id"], stage, k, v)
        st.success("حُفظ.")

with tab3:
    allp = pd.DataFrame(list_pain(None))
    if len(allp) < 5:
        st.info("يلزم 5 تسجيلات على الأقل.")
    else:
        for col, lab in (("angle_deg", "الزاوية"), ("emg_pct_mvc", "EMG % MVC")):
            d = allp[["nprs", col]].dropna()
            if len(d) >= 5:
                r, p = stats.spearmanr(d["nprs"], d[col])
                st.write(f"ارتباط سبيرمان بين الألم و{lab}: ρ = {r:.2f} (p = {p:.3f}, n = {len(d)})")
                st.scatter_chart(d, x=col, y="nprs")
