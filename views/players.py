# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from db import add_subject, delete_subject, list_subjects
from utils.constants import GROUPS
from utils.randomize import permuted_block_sequence
from utils.ui import setup

setup("اللاعبون والمجموعات", "📋")
st.caption("تسجيل العينة وتوزيعها على المجموعات الثلاث، وبيانات التكافؤ القبلي (العمر، BMI، مدة الإصابة، سنوات التدريب).")

with st.expander("🎲 مولّد التوزيع العشوائي (كتل متبدلة) — يُنفَّذ من شخص غير مشارك في القياس", expanded=False):
    c1, c2, c3 = st.columns(3)
    n = c1.number_input("عدد المشاركين المخطَّط", 3, 300, 54)
    block = c2.selectbox("حجم الكتلة", [3, 6, 9], index=1)
    seed = c3.number_input("البذرة (Seed) — يحتفظ بها المنفِّذ سرًّا", 0, 10**6, 2026)
    strata = st.text_input("الطبقات (اختياري، مفصولة بفاصلة)، مثال: مصارعة، جودو، تايكوندو", "")
    if st.button("توليد التسلسل"):
        names = [s.strip() for s in strata.split(",") if s.strip()] or ["كل المشاركين"]
        rows = []
        for i, nm in enumerate(names):
            seq = permuted_block_sequence(int(n), int(block), int(seed) + i)
            rows += [{"الطبقة": nm, "رقم المشارك": j + 1, "المجموعة": g} for j, g in enumerate(seq)]
        df_seq = pd.DataFrame(rows)
        st.dataframe(df_seq, use_container_width=True)
        st.download_button("⬇️ تنزيل التسلسل (CSV)", df_seq.to_csv(index=False).encode("utf-8-sig"), "allocation.csv")

st.subheader("إضافة لاعب")
with st.form("add_subject_form", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns(4)
    code = c1.text_input("الرمز (اختياري)", placeholder="P001")
    name = c2.text_input("الاسم")
    group = c3.selectbox("المجموعة", GROUPS)
    sport = c4.text_input("الرياضة القتالية")
    c5, c6, c7, c8, c9 = st.columns(5)
    age = c5.number_input("العمر", 16, 60, 22)
    height = c6.number_input("الطول (سم)", 120.0, 220.0, 175.0)
    weight = c7.number_input("الوزن (كغم)", 40.0, 160.0, 75.0)
    inj_days = c8.number_input("مدة الإصابة (يوم)", 0, 365, 30)
    tr_years = c9.number_input("سنوات التدريب", 0.0, 40.0, 4.0)
    note = st.text_area("ملاحظات الإصابة / التشخيص السريري")
    if st.form_submit_button("إضافة"):
        if not name.strip():
            st.error("أدخل اسم اللاعب.")
        else:
            add_subject(name.strip(), group, int(age), sport.strip(), note.strip(), float(height), float(weight),
                        int(inj_days), float(tr_years), code.strip() or None)
            st.success(f"أُضيف: {name}")
            st.rerun()

st.divider()
subjects = list_subjects()
if not subjects:
    st.info("لا يوجد لاعبون بعد.")
else:
    df = pd.DataFrame(subjects)
    cols = st.columns(len(GROUPS))
    for c, g in zip(cols, GROUPS):
        c.metric(g, int((df["group_name"] == g).sum()))
    st.dataframe(df.drop(columns=["created_at"], errors="ignore"), use_container_width=True)
    with st.expander("حذف لاعب"):
        opts = {f'#{s["id"]} {s["name"]}': s["id"] for s in subjects}
        pick = st.selectbox("اختر", list(opts.keys()))
        if st.button("حذف نهائي (يحذف قياساته أيضًا)", type="primary"):
            delete_subject(opts[pick])
            st.rerun()
