# -*- coding: utf-8 -*-
import os
import tempfile
import threading
import time

import pandas as pd
import streamlit as st

from db import add_coach_session, list_coach_sessions, list_subjects
from ml.live_coach import ExerciseCoach, LiveSession, process_video
from utils.constants import EXERCISE_BY_ID, level_of_week
from utils.ui import setup, subject_picker

setup("المدرّب الحركي المباشر (رؤية حاسوبية)", "🎥")
st.caption("الكاميرا تتبّع رأسك وكتفيك (BlazePose)، وتعدّ التكرارات وتعطي تغذية راجعة فورية: المدى، السرعة، التعويض، الحد الآمن. "
           "الزوايا من كاميرا 2D تقريبية (خطأ قد يبلغ 5–10°) ولا تغني عن القياس المرجعي. "
           "الثني والميل أدق من الدوران. التمارين الإيزومترية لا تُتتبَّع بالكاميرا.")

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    HAVE_WEBRTC = True
except Exception:
    HAVE_WEBRTC = False

CAM_EX = {i: e for i, e in EXERCISE_BY_ID.items() if "cam" in e}
subjects = list_subjects()
c1, c2, c3, c4 = st.columns(4)
with c1:
    subj = subject_picker(subjects, allow_none=True) if subjects else None
week = c2.number_input("الأسبوع", 1, 8, 1)
sess = c3.number_input("الجلسة", 1, 3, 1)
exid = c4.selectbox("التمرين", list(CAM_EX), format_func=lambda i: f"{i}. {CAM_EX[i]['name']}")
spec = dict(CAM_EX[exid]["cam"])
with st.expander("⚙️ إعدادات المدى المستهدف (يعدّلها الأخصائي بحسب مدى اللاعب)"):
    c = st.columns(3)
    spec["target"] = c[0].number_input("المدى المستهدف (°)", 5.0, 120.0, float(spec["target"]))
    spec["safe"] = c[1].number_input("الحد الآمن (°)", 10.0, 150.0, float(spec["safe"]))
    spec["min_rep_s"] = c[2].number_input("أقل زمن للتكرار (ث)", 1.0, 10.0, float(spec.get("min_rep_s", 3.0)))
view = {"pitch": "جانبية", "roll": "أمامية", "yaw": "أمامية", "multi": "أمامية (أو جانبية للثني/البسط)"}[spec["axis"]]
st.info(f"زاوية الكاميرا المناسبة: **{view}**. قف أو استلقِ في وضع محايد ثابت لثانيتين للمعايرة ثم ابدأ.")


def show(snap, ph):
    color = "red" if any("توقّف" in m for m in snap["messages"]) else "green"
    ph.markdown(f"### التكرارات: {snap['reps']} — الزاوية: {snap['rel']:.0f}° / الهدف {snap['target']:.0f}°\n\n"
                + "\n".join(f":{color}[**{m}**]" for m in snap["messages"]))


def save_summary(s):
    if not subj:
        st.warning("اختر لاعبًا للحفظ.")
        return
    add_coach_session(subject_id=subj["id"], week=int(week), session_no=int(sess), exercise_id=int(exid),
                      reps=s["reps"], mean_rom=s["mean_rom"], max_velocity=s["max_velocity"], too_fast=s["too_fast"],
                      short_rom=s["short_rom"], compensation=s["compensation"], over_limit=s["over_limit"],
                      quality_score=s["quality_score"])
    st.success("حُفظت الجلسة.")


tab1, tab2, tab3 = st.tabs(["بث مباشر", "فيديو مسجَّل", "سجل المدرّب"])

with tab1:
    if not HAVE_WEBRTC:
        st.warning("المكتبتان streamlit-webrtc وav غير مثبتتين. أضفهما إلى requirements.txt (موجودتان في النسخة المرفقة). "
                   "استخدم تبويب «فيديو مسجَّل» في الأثناء.")
    else:
        class Proc(VideoProcessorBase):
            def __init__(self):
                self.lock = threading.Lock()
                self.session = None
                self.spec = spec
                self.t0 = None
                self.est = None
                self.err = None

            def reset(self, new_spec):
                with self.lock:
                    self.spec = new_spec
                    self.session = None
                    self.t0 = None

            def recv(self, frame):
                img = frame.to_ndarray(format="rgb24")[:, ::-1].copy()  # مرآة
                try:
                    with self.lock:
                        if self.est is None:
                            from ml.neck_pose import PoseEstimator
                            self.est = PoseEstimator()
                        if self.session is None:
                            self.session = LiveSession(self.est, self.spec)
                            self.t0 = time.time()
                        img = self.session.process(img, time.time() - self.t0)
                except Exception as e:  # لا نُسقط البث عند خطأ في النموذج
                    self.err = str(e)
                return av.VideoFrame.from_ndarray(img, format="rgb24")

            def snapshot(self):
                with self.lock:
                    return self.session.last if self.session else None

            def summary(self):
                with self.lock:
                    return self.session.coach.summary() if self.session else None

        ctx = webrtc_streamer(key="coach", video_processor_factory=Proc,
                              rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                              media_stream_constraints={"video": True, "audio": False}, async_processing=True)
        b1, b2 = st.columns(2)
        if ctx.video_processor:
            if b1.button("🔄 بدء تمرين جديد / إعادة المعايرة"):
                ctx.video_processor.reset(spec)
            if b2.button("💾 إنهاء وحفظ ملخص التمرين"):
                s = ctx.video_processor.summary()
                if s:
                    save_summary(s)
                    st.json(s)
            if ctx.video_processor.err:
                st.error(f"تعذّر تشغيل نموذج الوضعية: {ctx.video_processor.err}")
            ph = st.empty()
            while ctx.state.playing:
                snap = ctx.video_processor.snapshot()
                if snap:
                    show(snap, ph)
                time.sleep(0.3)

with tab2:
    up = st.file_uploader("فيديو للتمرين (mp4/mov)", type=["mp4", "mov", "avi", "m4v"])
    if up is not None and st.button("▶️ تحليل الفيديو"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as f:
            f.write(up.read())
            path = f.name
        try:
            from ml.neck_pose import PoseEstimator
            bar = st.progress(0.0)
            s, last, tl = process_video(path, spec, PoseEstimator(), stride=2, progress=bar.progress)
            st.session_state["vid_sum"] = (s, tl)
            if last is not None:
                st.image(last, caption="آخر إطار محلَّل")
        except Exception as e:
            st.error(f"تعذّر التحليل: {e}")
        finally:
            os.unlink(path)
    if "vid_sum" in st.session_state:
        s, tl = st.session_state["vid_sum"]
        st.json(s)
        if tl:
            st.line_chart(pd.DataFrame(tl).set_index("t")["angle"])
        if st.button("💾 حفظ ملخص الفيديو"):
            save_summary(s)

with tab3:
    rows = list_coach_sessions(subj["id"] if subj else None)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.line_chart(df.sort_values("id").set_index("id")["quality_score"])
    else:
        st.caption("لا توجد جلسات محفوظة.")
