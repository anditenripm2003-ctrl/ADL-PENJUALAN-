"""Dashboard Streamlit Ruang Sales — membaca hasil model dari Colab (folder hasil_model/)."""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Ruang Sales · Klasterisasi FastDTW", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

from scripts import dashboard_model as dm  # noqa: E402

RENDER = [dm.page_overview, dm.page_exploration, dm.page_distance, dm.page_clustering, dm.page_recommendation, dm.page_methodology]
SLUGS = ["ringkasan", "eksplorasi", "jarak", "klasterisasi", "rekomendasi", "metodologi"]
pages = {name: st.Page(fn, title=name, url_path=slug, default=(i == 0)) for i, (name, fn, slug) in enumerate(zip(dm.PAGES, RENDER, SLUGS))}
st.session_state["_pages"] = pages
current = st.navigation(list(pages.values()), position="hidden")
dm.apply_css()
with st.sidebar:
    dm.sidebar(pages, current.title)
current.run()
st.markdown('<div class="footer">RUANG SALES · FastDTW · 3 representasi · 6 algoritma · hasil deskriptif, bukan prediksi.</div>', unsafe_allow_html=True)
