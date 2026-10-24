# -*- coding: utf-8 -*-
"""AI-NeckRehab: نظام متكامل لتأهيل شدّ عضلات الرقبة لدى لاعبي الفنون القتالية (أداة مساندة لأطروحة)."""
import streamlit as st

st.set_page_config(page_title="AI-NeckRehab", page_icon="🧠", layout="wide")

from utils.ui import RTL_CSS  # noqa: E402


def home():
    st.markdown(RTL_CSS, unsafe_allow_html=True)
    st.title("🧠 AI-NeckRehab")
    st.markdown("""
### نظام تأهيل ذكي متكامل لعضلات الرقبة
**نوع الذكاء الاصطناعي: تعلّم عميق موجَّه (Supervised Deep Learning) بنموذج هجين + رؤية حاسوبية:**

| المكوّن | النوع | الوظيفة |
|---|---|---|
| M1 EMG-Net | شبكة CNN أحادية البعد | تصنيف جودة النشاط العضلي |
| M2 | BlazePose (CNN مدرَّبة مسبقًا) + هندسة | قياس المدى، والمدرّب الحركي المباشر |
| M3 Dose-Net | شبكة MLP | اقتراح المقاومة والتكرارات والمجاميع |
| M4 الأمان | قواعد سريرية (ليست شبكة عمدًا) | تقييد مخرجات الشبكات |

**المجموعات:** G1 تجريبية (AI كامل) — G2 جرعة ثابتة بالجهاز نفسه — G3 تقليدية. **القياسات:** T0 / T1 / T2 / T3.

**أين أبدأ؟** من قسم «إدارة الدراسة» (اللاعبون ← القياسات ← البرنامج)، ثم «التمرين والمتابعة» أثناء الجلسات.
قسما «القياسات الخاصة» و«التحليل والإعداد» للاختبارات المتقدمة والإحصاء والنسخ الاحتياطي.
""")
    st.warning("النماذج الافتراضية مدرَّبة على بيانات تركيبية للعرض فقط؛ درّبها على بيانات حقيقية موسومة قبل أي استشهاد بنتائجها. "
               "وحدود مؤشر NMCI مبدئية وتُحدَّد من مجموعة ضابطة أو الأدبيات.")
    st.info("على Streamlit Cloud التخزين مؤقت: نزّل نسخة احتياطية بعد كل جلسة عمل (قسم «التحليل والإعداد»).")


def P(path, title, icon, slug):
    return st.Page(path, title=title, icon=icon, url_path=slug)


pages = {
    "": [st.Page(home, title="الرئيسية", icon="🏠", url_path="home", default=True)],
    "إدارة الدراسة": [
        P("views/players.py", "اللاعبون والمجموعات", "📋", "players"),
        P("views/measurements.py", "القياسات T0–T3", "🧪", "measurements"),
        P("views/program.py", "البرنامج والتمارين", "📚", "program"),
        P("views/report.py", "تقرير اللاعب", "📄", "report"),
    ],
    "التمرين والمتابعة": [
        P("views/coach.py", "المدرّب الحركي المباشر", "🎥", "coach"),
        P("views/ai_session.py", "جلسة التأهيل الذكية", "🤖", "ai-session"),
        P("views/pain.py", "الألم اللحظي وPPT", "🩺", "pain"),
    ],
    "القياسات الخاصة": [
        P("views/nmci.py", "مؤشر NMCI (التوازن والتحكم)", "⚖️", "nmci"),
        P("views/rom_camera.py", "مدى الحركة بالكاميرا", "📷", "rom"),
    ],
    "التحليل والإعداد": [
        P("views/emg_models.py", "EMG والنماذج العصبية", "📈", "emg-models"),
        P("views/stats.py", "الإحصاء", "📊", "stats"),
        P("views/backup.py", "النسخ الاحتياطي", "💾", "backup"),
    ],
}
st.navigation(pages).run()
