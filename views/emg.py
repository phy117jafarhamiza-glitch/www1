# -*- coding: utf-8 -*-
import io

import numpy as np
import pandas as pd
import streamlit as st

from db import add_emg_analysis, list_emg_analyses, list_subjects, update_baseline_mvc
from ml.emg_cnn import get_classifier
from ml.signal_processing import full_pipeline
from utils.constants import EMG_MUSCLES, STAGES
from utils.ui import setup, subject_picker, synthetic_warning

setup("تحليل EMG بالشبكة العصبية (EMG-Net)", "📈")
st.caption("M1: شبكة 1D-CNN تصنّف نوافذ EMG إلى: ضعيف/غير متوازن، طبيعي، مفرط/تعويضي. "
           "ملاحظة: EMG السطحي لا يسجّل المثنيات العميقة مباشرة (تُقيَّم بـ CCFT).")

clf = get_classifier()
st.info(f"النموذج: {clf.backend}")
synthetic_warning(clf.is_synthetic, "EMG-Net")

subjects = list_subjects()
c1, c2, c3 = st.columns(3)
with c1:
    subj = subject_picker(subjects, allow_none=True) if subjects else None
with c2:
    stage = st.selectbox("نقطة القياس", STAGES)
with c3:
    muscle = st.selectbox("العضلة", EMG_MUSCLES)

c4, c5 = st.columns(2)
fs = c4.number_input("تردد العينة fs (Hz)", 250, 4000, 1000, step=50)
mvc_ref = c5.number_input("RMS عند أقصى انقباض إرادي MVC (اتركه 0 لتقدير تقريبي)", 0.0, 1e6,
                          float(subj["baseline_mvc"]) if subj and subj.get("baseline_mvc") else 0.0)

up = st.file_uploader("ملف CSV (عمود واحد للإشارة الخام، أو عمود باسم signal)", type=["csv", "txt"])
demo = st.checkbox("استخدام إشارة تجريبية")

raw = None
src = None
if up is not None:
    try:
        df = pd.read_csv(up)
        col = "signal" if "signal" in df.columns else df.select_dtypes("number").columns[0]
        raw = df[col].to_numpy(float)
        src = up.name
    except Exception as e:
        st.error(f"تعذّرت قراءة الملف: {e}")
elif demo:
    t = np.arange(0, 6, 1 / fs)
    rng = np.random.default_rng(1)
    raw = rng.standard_normal(len(t)) * (0.2 + 0.5 * (np.sin(2 * np.pi * 0.4 * t) > 0)) * 50
    src = "demo"

if raw is not None and len(raw) > fs:
    out = full_pipeline(raw, fs, ref_amp=mvc_ref if mvc_ref > 0 else None)
    st.line_chart(pd.DataFrame({"مرشّحة": out["filtered"][::max(1, len(raw) // 2000)]}))
    results, majority, q_normal = clf.predict_windows(out["raw_windows"], out["segments"])
    if not results:
        st.warning("لا توجد نوافذ كافية.")
    else:
        if mvc_ref <= 0:
            st.warning("لم تُدخل MVC: التطبيع تقريبي (المئين 95)، ولا يصح الاعتماد عليه سريريًا.")
        st.subheader(f"الحكم الغالب: {majority}  —  احتمال «طبيعي» المتوسط: {q_normal:.2f}")
        wdf = pd.DataFrame([{"نافذة": i, "التصنيف": r["label"], "الثقة": round(r["confidence"], 3),
                             **{k: round(v, 4) for k, v in r["features"].items()}} for i, r in enumerate(results)])
        st.dataframe(wdf, use_container_width=True)
        st.download_button("⬇️ تنزيل النوافذ للوسم الخبير (CSV)",
                           pd.DataFrame(out["raw_windows"]).assign(label="").to_csv(index=False).encode("utf-8-sig"),
                           "emg_windows_for_labeling.csv")
        cc1, cc2 = st.columns(2)
        if cc1.button("💾 حفظ التحليل") and subj:
            mean_f = {k: float(np.mean([r["features"].get(k, 0) for r in results])) for k in ["rms", "mav", "zc", "wl"]}
            conf = float(np.mean([r["confidence"] for r in results]))
            add_emg_analysis(subj["id"], stage, muscle, src, majority, conf, mean_f, clf.backend)
            st.success("حُفظ.")
        if subj and cc2.button("تعيين RMS هذا كـ MVC الأساسية للاعب"):
            update_baseline_mvc(subj["id"], float(out["ref_amp"]))
            st.success("حُدّثت MVC.")
        if not subj:
            st.caption("اختر لاعبًا لتفعيل الحفظ.")

st.divider()
st.subheader("التحليلات المحفوظة")
rows = list_emg_analyses(subj["id"] if subj else None)
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.caption("لا توجد تحليلات.")
