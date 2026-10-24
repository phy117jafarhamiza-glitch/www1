# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from db import get_measurements, list_subjects
from utils import stats as S
from utils.constants import COVARIATES, PRIMARY_KEY, STAGES, TEST_META, TEST_NAMES
from utils.ui import setup

setup("التحليل الإحصائي", "📊")
st.caption("اختبارات لا معلمية عند عدم الاعتدالية، تصحيح Holm، ANCOVA بتعديل القياس القبلي، حجم الأثر، والمستجيبون (MCID). "
           "للتحليل النهائي (ANOVA مختلط/LMM وتعويض القيم المفقودة) استخدم SPSS/R عبر تصدير Excel.")

subjects, meas = list_subjects(), get_measurements()
if len(subjects) < 3 or not meas:
    st.info("أضف لاعبين وقياسات أولًا.")
    st.stop()

tab0, tab1 = st.tabs(["تكافؤ المجموعات", "تحليل متغير"])
with tab0:
    eq = S.baseline_equivalence(subjects, COVARIATES)
    st.dataframe(eq, use_container_width=True) if len(eq) else st.info("بيانات غير كافية.")

with tab1:
    key = st.selectbox("المتغير", list(TEST_META.keys()), format_func=lambda k: TEST_NAMES[k],
                       index=list(TEST_META.keys()).index(PRIMARY_KEY))
    df = S.wide_table(meas, subjects, key, STAGES)
    st.dataframe(df, use_container_width=True)
    st.subheader("الإحصاء الوصفي")
    st.dataframe(S.descriptive(df, STAGES), use_container_width=True)
    st.subheader("اختبار الاعتدالية (Shapiro–Wilk)")
    st.dataframe(S.normality(df, STAGES), use_container_width=True)

    st.subheader("التغيّر داخل كل مجموعة (مقابل T0)")
    w = S.within_group_change(df, STAGES[0], STAGES[1:])
    st.dataframe(w, use_container_width=True) if len(w) else st.caption("بيانات غير كافية.")

    st.subheader("مقارنة المجموعات")
    df = df.copy()
    for s_ in STAGES[1:]:
        df[f"Δ {s_}"] = df[s_] - df[STAGES[0]]
    col = st.selectbox("عمود المقارنة", [f"Δ {s_}" for s_ in STAGES[1:]] + STAGES)
    r = S.between_groups(df, col)
    if r.get("ok"):
        st.write(f"**{r['test']}** — p = {r['p']:.4f} — الأثر: {r['effect']} — الاعتدالية: {'نعم' if r['normal'] else 'لا'}")
        st.dataframe(pd.DataFrame(r["pairs"]), use_container_width=True)
    else:
        st.caption(r.get("reason"))

    st.subheader("ANCOVA (البعدي ~ المجموعة + القبلي)")
    post = st.selectbox("القياس البعدي", STAGES[1:], index=1)
    a = S.ancova(df, STAGES[0], post)
    if a.get("ok"):
        st.write(f"F({a['df'][0]},{a['df'][1]}) = {a['F']:.3f}, p = {a['p']:.4f}, η²جزئي = {a['partial_eta2']:.3f}")
        st.write({k: round(v, 2) for k, v in a["adjusted_means"].items()})
    else:
        st.caption(a.get("reason"))

    if key.startswith("nprs"):
        st.subheader("المستجيبون (انخفاض ≥ 2 نقطة أو ≥ 30%)")
        out, p = S.responders(df, STAGES[0], post)
        st.dataframe(out, use_container_width=True)
        if p is not None:
            st.write(f"p = {p:.4f}")

    chart = df.groupby("group_name")[STAGES].mean().T
    st.line_chart(chart)
    st.download_button("⬇️ الجدول (CSV)", df.to_csv(index=False).encode("utf-8-sig"), f"{key}.csv")
