# -*- coding: utf-8 -*-
import streamlit as st

from db import db_bytes, export_excel_bytes, restore_db
from utils.ui import setup

setup("النسخ الاحتياطي والتصدير", "💾")
st.error("⚠️ على Streamlit Community Cloud يكون نظام الملفات مؤقتًا: قد تُفقد البيانات عند إعادة تشغيل التطبيق. "
         "نزّل نسخة احتياطية بعد كل جلسة عمل، وأعد رفعها عند الحاجة.")
c1, c2 = st.columns(2)
c1.download_button("⬇️ نسخة قاعدة البيانات (.db)", db_bytes(), "neck_rehab_backup.db")
c2.download_button("⬇️ تصدير Excel (متضمنًا ورقة SPSS)", export_excel_bytes(), "neck_rehab_export.xlsx")
st.divider()
up = st.file_uploader("استعادة من ملف .db", type=["db"])
if up is not None and st.button("استعادة (تستبدل البيانات الحالية)", type="primary"):
    restore_db(up.read())
    st.success("تمت الاستعادة.")
