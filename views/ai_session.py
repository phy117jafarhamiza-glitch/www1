# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from db import add_adverse_event, add_ai_set, last_ai_set, list_adverse_events, list_ai_sets, list_subjects
from ml.dose_net import get_dosenet
from ml.emg_cnn import get_classifier
from ml.safety import SYMPTOMS, evaluate
from ml.signal_processing import full_pipeline
from utils.constants import (DIRECTION_AR, DIRECTIONS, EMG_QUALITY_LABELS, FIXED_DOSE, GROUP_AI, GROUP_CONV,
                             GROUP_FIXED, PHASE_NAMES, SESSION_MINUTES, WEEK_RESISTANCE_CAP, phase_of_week)
from utils.ui import setup, subject_picker, synthetic_warning

setup("جلسة التأهيل الذكية", "🤖")
st.caption("خط الأنابيب: EMG-Net (M1) + الكاميرا (M2) ⟶ Dose-Net (M3) ⟶ طبقة الأمان (M4) ⟶ الأخصائي يوافق أو يعدّل (Human-in-the-loop).")

subjects = list_subjects()
if not subjects:
    st.warning("أضف لاعبين أولًا.")
    st.stop()

subj = subject_picker(subjects)
c1, c2, c3 = st.columns(3)
week = c1.number_input("الأسبوع", 1, 8, 1)
session_no = c2.number_input("رقم الجلسة", 1, 40, 1)
st.info(f"{PHASE_NAMES[phase_of_week(int(week))]} — مدة الجلسة المقترحة {SESSION_MINUTES[int(week)]} دقيقة — سقف المقاومة {WEEK_RESISTANCE_CAP[int(week)]}% MVC")

g = subj["group_name"]
tab1, tab2 = st.tabs(["الجلسة", "الأحداث الضارة والسجل"])

with tab1:
    if g == GROUP_CONV:
        st.info("هذا اللاعب في المجموعة التقليدية (G3): لا يستخدم الجهاز ولا النظام. سجّل الحضور في الأحداث/السجل إن لزم.")
    elif g == GROUP_FIXED:
        fd = FIXED_DOSE[int(week)]
        st.success(f"G2 — جرعة ثابتة بالجهاز نفسه: {fd['sets']} مجاميع × {fd['reps']} تكرار، راحة {fd['rest_sets']} ث. "
                   "لا يوصي النظام الذكي بأي تعديل لهذه المجموعة.")
    else:
        clf, dn = get_classifier(), get_dosenet()
        synthetic_warning(clf.is_synthetic or dn.is_synthetic, "EMG-Net/Dose-Net")
        prev = last_ai_set(subj["id"])
        d = st.selectbox("اتجاه التمرين", DIRECTIONS, format_func=lambda x: DIRECTION_AR[x])
        set_no = st.number_input("رقم المجموعة", 1, 6, 1)

        st.markdown("**1) جودة النشاط العضلي (EMG-Net)** — ارفع إشارة المجموعة أو أدخل القيمة يدويًا")
        up = st.file_uploader("إشارة EMG للمجموعة (CSV)", type=["csv"])
        fs = st.number_input("fs (Hz)", 250, 4000, 1000, step=50)
        mvc = st.number_input("RMS عند MVC للاعب", 0.0, 1e6, float(subj.get("baseline_mvc") or 0.0))
        qn, label, emg_pct = 0.6, EMG_QUALITY_LABELS[1], 40.0
        if up is not None:
            try:
                df = pd.read_csv(up)
                col = "signal" if "signal" in df.columns else df.select_dtypes("number").columns[0]
                out = full_pipeline(df[col].to_numpy(float), fs, ref_amp=mvc if mvc > 0 else None)
                res, label, qn = clf.predict_windows(out["raw_windows"], out["segments"])
                emg_pct = float(min(200.0, 100.0 * out["envelope"].mean() / out["ref_amp"]))
                st.write(f"التصنيف: **{label}** — احتمال طبيعي {qn:.2f} — EMG ≈ {emg_pct:.0f}% MVC")
            except Exception as e:
                st.error(f"تعذّرت المعالجة: {e}")
        else:
            qn = st.slider("احتمال «طبيعي»", 0.0, 1.0, 0.6)
            emg_pct = st.number_input("EMG % MVC", 0.0, 200.0, 40.0)
            label = st.selectbox("تصنيف الجودة", EMG_QUALITY_LABELS, index=1)

        st.markdown("**2) الحركة والألم والأعراض**")
        c = st.columns(4)
        ang = c[0].number_input("الزاوية % من الحد الآمن", 0.0, 150.0, 50.0)
        vel = c[1].number_input("السرعة (°/ث)", 0.0, 200.0, 40.0)
        nprs = c[2].number_input("الألم الآن (NPRS)", 0.0, 10.0, 2.0)
        nprs0 = c[3].number_input("الألم بداية الجلسة", 0.0, 10.0, 2.0)
        asym = st.number_input("تفاوت يمين/يسار % (إن توفّر)", 0.0, 100.0, 0.0)
        symptoms = st.multiselect("أعراض خطرة", SYMPTOMS)
        prev_res = st.number_input("المقاومة السابقة %MVC", 0.0, 100.0, float(prev["final_resistance"]) if prev and prev.get("final_resistance") is not None else 15.0)

        streak = 0
        for r in list_ai_sets(subj["id"], 5):
            if r.get("quality_label") == EMG_QUALITY_LABELS[2]:
                streak += 1
            else:
                break
        if label == EMG_QUALITY_LABELS[2]:
            streak += 1

        if st.button("⚙️ احسب التوصية"):
            state = dict(direction=d, quality_normal=qn, emg_pct_mvc=emg_pct, head_angle_pct=ang, velocity=vel,
                         nprs_now=nprs, week=int(week), prev_resistance=prev_res)
            prop = dn.predict(state)
            safe = evaluate(prop, dict(prev_resistance=prev_res, week=int(week), nprs_now=nprs, nprs_session_start=nprs0,
                                       head_angle_pct=ang, symptoms=symptoms, excess_streak=streak, asymmetry_pct=asym))
            st.session_state["rec"] = dict(state=state, prop=prop, safe=safe, label=label, nprs0=nprs0)

        rec = st.session_state.get("rec")
        if rec:
            p, s = rec["prop"], rec["safe"]
            cc = st.columns(2)
            cc[0].metric("مقترح Dose-Net", f"{p['resistance']:.0f}% MVC", f"{p['sets']}×{p['reps']}")
            cc[1].metric("بعد طبقة الأمان", f"{s['resistance']:.0f}% MVC", f"{s['sets']}×{s['reps']}")
            if s["action"] in ("stop_set", "stop_session"):
                st.error("⛔ " + " ".join(s["messages"]))
            for m in s["messages"] if s["action"] not in ("stop_set", "stop_session") else []:
                st.warning(m)
            if s["alert_physio"]:
                st.error("🔔 تنبيه للأخصائي: يلزم مراجعة سريرية.")
            if st.button("✅ اعتماد وتسجيل (موافقة الأخصائي)"):
                stt = rec["state"]
                add_ai_set(subject_id=subj["id"], week=int(week), session_no=int(session_no), set_no=int(set_no),
                           direction=d, emg_pct_mvc=stt["emg_pct_mvc"], quality_normal=stt["quality_normal"],
                           quality_label=rec["label"], head_angle_pct=stt["head_angle_pct"], velocity=stt["velocity"],
                           nprs_now=stt["nprs_now"], nprs_session_start=rec["nprs0"], prev_resistance=stt["prev_resistance"],
                           rec_resistance=p["resistance"], rec_reps=p["reps"], rec_sets=p["sets"],
                           safety_action=s["action"], safety_messages=" | ".join(s["messages"]),
                           final_resistance=s["resistance"])
                st.session_state.pop("rec", None)
                st.success("سُجّلت المجموعة.")
                st.rerun()

with tab2:
    with st.form("ae"):
        desc = st.text_area("وصف الحدث الضار / الانسحاب")
        sev = st.selectbox("الشدة", ["خفيف", "متوسط", "شديد"])
        if st.form_submit_button("تسجيل") and desc.strip():
            add_adverse_event(subj["id"], int(week), desc.strip(), sev)
            st.success("سُجّل.")
    ev = list_adverse_events(subj["id"])
    if ev:
        st.dataframe(pd.DataFrame(ev), use_container_width=True)
    sets = list_ai_sets(subj["id"])
    if sets:
        st.subheader("سجل مجاميع الذكاء الاصطناعي")
        sdf = pd.DataFrame(sets)
        st.dataframe(sdf, use_container_width=True)
        st.download_button("⬇️ CSV", sdf.to_csv(index=False).encode("utf-8-sig"), "ai_sets.csv")
