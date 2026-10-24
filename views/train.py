# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import cohen_kappa_score

from db import add_dose_label, list_dose_labels
from ml.dose_net import evaluate_against_experts, get_dosenet, reload_dosenet, train_from_expert_labels
from ml.emg_cnn import INPUT_LEN, cross_validate_subjectwise, get_classifier, load_meta, reload_classifier, train_and_save
from utils.constants import DIRECTIONS, DIRECTION_AR, EMG_QUALITY_LABELS
from utils.ui import setup

setup("تدريب والتحقق من النماذج", "🧠")
st.caption("تدريب الشبكات على بيانات حقيقية موسومة من الخبراء، تحقق بحسب اللاعب (Subject-wise)، ثم تجميد النموذج قبل بدء التجربة.")

tab1, tab2, tab3 = st.tabs(["EMG-Net (CNN)", "Dose-Net (MLP)", "حالة النماذج"])

with tab1:
    st.markdown("**ملف النوافذ الموسومة (CSV):** 256 عمودًا للعينات + `label` (0 ضعيف، 1 طبيعي، 2 مفرط) + `subject` (رمز اللاعب).")
    up = st.file_uploader("ملف النوافذ الموسومة", type=["csv"], key="lab")
    epochs = st.slider("عدد الدورات (epochs)", 5, 80, 25)
    if up is not None:
        df = pd.read_csv(up)
        need = {"label", "subject"}
        sig_cols = [c for c in df.columns if c not in need]
        if not need.issubset(df.columns) or len(sig_cols) != INPUT_LEN:
            st.error(f"الصيغة غير صحيحة: يلزم {INPUT_LEN} عمودًا للإشارة + label + subject.")
        else:
            df = df.dropna(subset=["label"])
            X = df[sig_cols].to_numpy(float)
            y = df["label"].astype(int).to_numpy()
            g = pd.factorize(df["subject"])[0]
            st.write(f"النوافذ: {len(y)} — اللاعبون: {len(np.unique(g))} — توزيع الفئات: {np.bincount(y, minlength=3).tolist()}")
            if st.button("▶️ تحقق متقاطع بحسب اللاعب"):
                with st.spinner("جارٍ التحقق..."):
                    res = cross_validate_subjectwise(X, y, g, n_splits=5, epochs=epochs)
                c = st.columns(3)
                c[0].metric("الدقة", f"{res['accuracy']:.3f}")
                c[1].metric("Macro-F1", f"{res['macro_f1']:.3f}")
                c[2].metric("Cohen κ", f"{res['kappa']:.3f}")
                st.write("مصفوفة الالتباس (صفوف = الحقيقي):")
                st.dataframe(pd.DataFrame(res["confusion_matrix"], index=EMG_QUALITY_LABELS, columns=EMG_QUALITY_LABELS))
                st.session_state["cv_res"] = res
            if st.button("🔒 تدريب نهائي وتجميد النموذج"):
                with st.spinner("جارٍ التدريب..."):
                    extra = {}
                    if "cv_res" in st.session_state:
                        r = st.session_state["cv_res"]
                        extra = {"cv_accuracy": r["accuracy"], "cv_macro_f1": r["macro_f1"], "cv_kappa": r["kappa"]}
                    train_and_save(X, y, epochs=epochs, source="real", extra_meta=extra)
                    reload_classifier()
                st.success("دُرّب النموذج وحُفظ. لا تعِد التدريب بعد بدء التجربة.")
    st.divider()
    st.markdown("**اتفاق مُقيِّمَين (Inter-rater κ):** ارفع ملفًا بعمودين rater1 وrater2.")
    up2 = st.file_uploader("ملف المقيّمين", type=["csv"], key="irr")
    if up2 is not None:
        d = pd.read_csv(up2).dropna()
        if {"rater1", "rater2"}.issubset(d.columns) and len(d) > 1:
            st.metric("Cohen κ بين المقيّمين", f"{cohen_kappa_score(d['rater1'], d['rater2']):.3f}")
        else:
            st.error("يلزم العمودان rater1 وrater2.")

with tab2:
    st.markdown("Dose-Net تتعلم من وصفات الأخصائيين (Imitation learning). لا تعلُّم تعزيزي على المصابين.")
    with st.form("dose_label"):
        c = st.columns(4)
        direction = c[0].selectbox("الاتجاه", DIRECTIONS, format_func=lambda d: DIRECTION_AR[d])
        qn = c[1].number_input("احتمال «طبيعي»", 0.0, 1.0, 0.6)
        emgp = c[2].number_input("EMG % MVC", 0.0, 200.0, 40.0)
        ang = c[3].number_input("الزاوية % من الحد الآمن", 0.0, 150.0, 50.0)
        c = st.columns(4)
        vel = c[0].number_input("السرعة (°/ث)", 0.0, 200.0, 40.0)
        nprs = c[1].number_input("الألم الآن", 0.0, 10.0, 2.0)
        week = c[2].number_input("الأسبوع", 1, 8, 1)
        prev = c[3].number_input("المقاومة السابقة %MVC", 0.0, 100.0, 20.0)
        c = st.columns(4)
        er = c[0].number_input("وصفة الخبير: المقاومة %", 0.0, 100.0, 25.0)
        ereps = c[1].number_input("التكرارات", 8, 15, 10)
        esets = c[2].number_input("المجاميع", 2, 4, 2)
        rater = c[3].text_input("المقيّم")
        if st.form_submit_button("إضافة وصفة خبير"):
            add_dose_label(dict(direction=direction, quality_normal=qn, emg_pct_mvc=emgp, head_angle_pct=ang,
                                velocity=vel, nprs_now=nprs, week=int(week), prev_resistance=prev,
                                exp_resistance=er, exp_reps=int(ereps), exp_sets=int(esets), rater=rater))
            st.success("أُضيفت.")
    up3 = st.file_uploader("أو رفع وصفات متعددة (CSV بنفس أسماء الأعمدة)", type=["csv"], key="dl")
    if up3 is not None and st.button("استيراد الوصفات"):
        d = pd.read_csv(up3)
        for r in d.to_dict("records"):
            add_dose_label(r)
        st.success(f"استُورد {len(d)} سجلًا.")
    labels = list_dose_labels()
    st.write(f"وصفات الخبراء المسجّلة: **{len(labels)}**")
    if labels:
        st.dataframe(pd.DataFrame(labels), use_container_width=True)
        if len(labels) >= 30:
            if st.button("🔒 تدريب Dose-Net على وصفات الخبراء"):
                train_from_expert_labels(labels)
                reload_dosenet()
                st.success("تم التدريب.")
            if st.button("قياس الاتفاق مع الخبراء"):
                ev = evaluate_against_experts(labels)
                st.json(ev)
                st.caption("تنبيه: القياس على نفس بيانات التدريب متفائل؛ استخدم مجموعة اختبار منفصلة للتقرير.")
        else:
            st.info("يلزم 30 وصفة على الأقل (يُفضّل 300+).")

with tab3:
    clf = get_classifier()
    st.write("**EMG-Net:**", clf.backend)
    st.json(load_meta())
    dn = get_dosenet()
    st.write("**Dose-Net:**", "تركيبي (للعرض فقط)" if dn.is_synthetic else f"مدرَّب على وصفات خبراء ({dn.meta.get('n')})")
