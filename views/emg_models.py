# -*- coding: utf-8 -*-
import os

import streamlit as st

from utils.ui import embed, setup

HERE = os.path.dirname(os.path.abspath(__file__))
setup("EMG والنماذج العصبية", "📈")
t1, t2 = st.tabs(["📈 تحليل EMG (EMG-Net)", "🧠 تدريب والتحقق من النماذج"])
with t1:
    embed(os.path.join(HERE, "emg.py"))
with t2:
    embed(os.path.join(HERE, "train.py"))
