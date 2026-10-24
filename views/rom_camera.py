# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from db import add_pose_rom, add_rom_validation, list_pose_rom, list_rom_validation, list_subjects
from utils.constants import DIRECTION_AR, DIRECTIONS, STAGES
from utils.ui import setup, subject_picker
from utils.validation import bland_altman, icc_2_1, interpret_icc

setup("مدى الحركة بالكاميرا (BlazePose)", "📷")
st.caption("M2: شبكة BlazePose (CNN مدرَّبة مسبقًا) تستخرج النقاط، ثم تُحسب زاوية الرقبة هندسيًا. "
           "يجب إثبات صلاحية القياس مقابل مقياس مرجعي قبل الاعتماد عليه (تبويب التحقق).")

KEYMAP = {"flex": "flexion", "ext": "extension", "lat_r": "lateral_r", "lat_l": "lateral_l",
          "rot_r": "rotation_r", "rot_l": "rotation_l"}

tab1, tab2 = st.tabs(["القياس", "التحقق من الصلاحية (ICC / Bland–Altman)"])

with tab1:
    subjects = list_subjects()
    c1, c2, c3 = st.columns(3)
    with c1:
        subj = subject_picker(subjects, allow_none=True) if subjects else None
    stage = c2.selectbox("نقطة القياس", STAGES)
    d = c3.selectbox("الاتجاه", DIRECTIONS, format_func=lambda x: DIRECTION_AR[x])
    st.write("التقط صورة الوضع المحايد ثم صورة الذروة (من الجانب لثني/بسط/ميل، ومن الأمام للدوران).")
    a = st.camera_input("الوضع المحايد", key="neutral")
    b = st.camera_input("وضع الذروة", key="peak")
    if a is not None and b is not None and st.button("حساب المدى"):
        try:
            from ml.neck_pose import PoseEstimator, rom_from_neutral_and_peak
            est = PoseEstimator()
            n_lm = est.estimate(np.array(Image.open(a).convert("RGB")))
            p_lm = est.estimate(np.array(Image.open(b).convert("RGB")))
            if n_lm is None or p_lm is None:
                st.error("لم يُكتشف الجسم في إحدى الصورتين.")
            else:
                st.session_state["rom_val"] = float(rom_from_neutral_and_peak(n_lm, p_lm, KEYMAP[d]))
        except Exception as e:
            st.error(f"تعذّر تشغيل MediaPipe: {e}")
    if "rom_val" in st.session_state:
        v = st.session_state["rom_val"]
        st.metric("المدى المقدَّر (درجة)", f"{v:.1f}")
        if subj and st.button("💾 حفظ"):
            add_pose_rom(subj["id"], stage, "camera", d, v)
            st.success("حُفظ. انقل القيمة المعتمدة إلى صفحة القياسات.")
    rows = list_pose_rom(subj["id"] if subj else None)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab2:
    st.markdown("أدخل قراءة الكاميرا وقراءة المقياس المرجعي (جونيومتر/إنكلينومتر) لنفس الحركة.")
    subjects = list_subjects()
    with st.form("val"):
        c = st.columns(4)
        sid = c[0].selectbox("اللاعب", [None] + [s["id"] for s in subjects],
                             format_func=lambda i: "—" if i is None else next(s["name"] for s in subjects if s["id"] == i))
        dd = c[1].selectbox("الاتجاه", DIRECTIONS, format_func=lambda x: DIRECTION_AR[x], key="vd")
        cam = c[2].number_input("الكاميرا (°)", 0.0, 150.0, 0.0)
        ref = c[3].number_input("المرجع (°)", 0.0, 150.0, 0.0)
        if st.form_submit_button("إضافة زوج قياسات"):
            add_rom_validation(sid, dd, cam, ref)
            st.success("أُضيف.")
    v = list_rom_validation()
    if len(v) >= 3:
        df = pd.DataFrame(v)
        icc = icc_2_1(df["camera_value"], df["reference_value"])
        ba = bland_altman(df["camera_value"], df["reference_value"])
        c = st.columns(4)
        c[0].metric("ICC(2,1)", f"{icc['icc']:.3f}" if icc["icc"] is not None else "—")
        c[1].metric("MAE (°)", f"{ba['mae']:.2f}")
        c[2].metric("الانحياز (°)", f"{ba['bias']:.2f}")
        c[3].metric("حدود الاتفاق", f"{ba['loa_low']:.1f} / {ba['loa_high']:.1f}")
        st.caption(f"تفسير ICC: {interpret_icc(icc['icc'])} — فاصل الثقة 95%: {icc['ci']}")
        st.scatter_chart(df.rename(columns={"reference_value": "المرجع", "camera_value": "الكاميرا"}), x="المرجع", y="الكاميرا")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("يلزم 3 أزواج على الأقل (يُفضّل 30+).")
