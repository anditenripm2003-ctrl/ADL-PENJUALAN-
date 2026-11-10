"""Dashboard Streamlit Ruang Sales — membaca hasil model dari Colab (folder hasil_model/)."""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Ruang Sales · Klasterisasi FastDTW", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Tema gelap dipaksa dari kode, jadi tetap gelap walau .streamlit/config.toml belum ada di repo.
_THEME = {"theme.base": "dark", "theme.primaryColor": "#00D5F2", "theme.backgroundColor": "#06172B",
          "theme.secondaryBackgroundColor": "#0B2945", "theme.textColor": "#EAF2FA"}
try:
    from streamlit import config as _stc
    _changed = False
    for _k, _v in _THEME.items():
        if _stc.get_option(_k) != _v:
            _stc.set_option(_k, _v)
            _changed = True
    if _changed and not st.session_state.get("_theme_set"):
        st.session_state["_theme_set"] = True
        st.rerun()
except Exception:  # st.rerun melempar pengecualian kontrol turunan BaseException, jadi tidak tertangkap di sini
    pass

from scripts import dashboard_model as dm  # noqa: E402

RENDER = [dm.page_overview, dm.page_exploration, dm.page_distance, dm.page_clustering, dm.page_recommendation, dm.page_methodology]
SLUGS = ["ringkasan", "eksplorasi", "jarak", "klasterisasi", "rekomendasi", "metodologi"]
pages = {name: st.Page(fn, title=name, url_path=slug, icon=f":material/{ic}:", default=(i == 0))
         for i, (name, fn, slug, ic) in enumerate(zip(dm.PAGES, RENDER, SLUGS, dm.ICONS))}
st.session_state["_pages"] = pages
current = st.navigation(list(pages.values()), position="hidden")  # navigasi 6 halaman ada di sidebar
dm.apply_css()
with st.sidebar:
    dm.sidebar(pages, current.title)
current.run()
st.markdown('<div class="footer">RUANG SALES · FastDTW · 3 representasi · 6 algoritma · hasil deskriptif dari data historis, bukan prediksi.</div>', unsafe_allow_html=True)
