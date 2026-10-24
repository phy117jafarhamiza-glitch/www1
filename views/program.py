# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from db import list_sessions, list_subjects, upsert_session
from utils.constants import (EXERCISES, FIXED_DOSE, LEVELS, SESSION_MINUTES, SESSION_SPLIT, SESSIONS_PER_WEEK,
                             WEEK_RESISTANCE_CAP, exercises_for_week, level_of_week)
from utils.ui import setup, subject_picker

setup("البرنامج التأهيلي وبنك التمارين", "📚")
st.caption("8 أسابيع × 3 جلسات، من 15 إلى 40 دقيقة، بتدرج من السهل إلى الصعب. القيم مقترحة وتُعتمد مع الخبراء (تحكيم CVR).")

weeks = []
for w in range(1, 9):
    wu, mn, cd = SESSION_SPLIT[w]
    fd = FIXED_DOSE[w]
    weeks.append({"الأسبوع": w, "المستوى": LEVELS[level_of_week(w)], "المدة (د)": SESSION_MINUTES[w],
                  "إحماء": wu, "رئيسي": mn, "ختام": cd, "المجاميع×التكرارات": f"{fd['sets']} × {fd['reps']}",
                  "راحة (ث)": fd["rest_sets"], "سقف المقاومة %MVC": WEEK_RESISTANCE_CAP[w]})
wdf = pd.DataFrame(weeks)
tab1, tab2, tab3 = st.tabs(["جدول الأسابيع", "خطة الجلسة", "سجل الحضور"])

with tab1:
    st.dataframe(wdf, use_container_width=True, hide_index=True)
    st.bar_chart(wdf.set_index("الأسبوع")["المدة (د)"])
    st.download_button("⬇️ الجدول (CSV)", wdf.to_csv(index=False).encode("utf-8-sig"), "program_weeks.csv")
    st.subheader("بنك التمارين (17)")
    edf = pd.DataFrame([{"#": e["id"], "التمرين": e["name"], "الهدف": e["aim"], "المستويات": "، ".join(str(l) for l in e["levels"]),
                         "الأسابيع": e["weeks"], "الجرعة": e["dose"], "التتبع": e["tracking"]} for e in EXERCISES])
    st.dataframe(edf, use_container_width=True, hide_index=True)

with tab2:
    w = st.selectbox("الأسبوع", list(range(1, 9)))
    wu, mn, cd = SESSION_SPLIT[w]
    exs = exercises_for_week(w)
    st.info(f"{LEVELS[level_of_week(w)]} — الجلسة {SESSION_MINUTES[w]} دقيقة = إحماء {wu} + رئيسي {mn} + ختام {cd}")
    per = mn / max(1, len([e for e in exs if e["id"] != 5]))
    plan = [{"#": e["id"], "التمرين": e["name"], "الجرعة": e["dose"], "التتبع": e["tracking"],
             "زمن تقريبي (د)": 0 if e["id"] == 5 else round(per, 1)} for e in exs]
    st.dataframe(pd.DataFrame(plan), use_container_width=True, hide_index=True)
    st.caption("تمرين الإطالة (5) ضمن الإحماء/الختام؛ يُوزَّع الزمن الرئيسي بالتساوي بين بقية التمارين، ويعدّله الأخصائي.")

with tab3:
    subjects = list_subjects()
    if not subjects:
        st.info("أضف لاعبين أولًا.")
    else:
        subj = subject_picker(subjects)
        c = st.columns(5)
        w = c[0].number_input("الأسبوع", 1, 8, 1, key="sw")
        s = c[1].number_input("الجلسة", 1, SESSIONS_PER_WEEK, 1, key="ss")
        mins = c[2].number_input("الدقائق الفعلية", 0.0, 120.0, float(SESSION_MINUTES[int(w)]))
        rpe = c[3].number_input("الجهد المدرك RPE (0–10)", 0.0, 10.0, None)
        done = c[4].checkbox("اكتملت الجلسة", True)
        if st.button("حفظ الجلسة"):
            upsert_session(subj["id"], int(w), int(s), mins, done, rpe)
            st.rerun()
        rows = list_sessions(subj["id"])
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df.drop(columns=["subject_id"]), use_container_width=True)
            st.metric("الالتزام (الجلسات المكتملة من 24)", f"{100 * df['completed'].sum() / 24:.0f}%")
