"""Dashboard Streamlit Ruang Sales — membaca hasil model dari notebook Colab (hasil_model/).

Tidak ada pelatihan ulang di dashboard: semua label, centroid, metrik, dan koordinat peta sudah
dihitung di Colab (FastDTW · 3 representasi · 6 algoritma). Alurnya tetap seperti R Shiny:
pengguna memilih representasi, algoritma, dan parameter, lalu menekan **Tampilkan hasil**.
"""
from __future__ import annotations

import io
import zipfile
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import adjusted_rand_score

import config
from scripts import interactive as ix
from scripts import interactive_viz as iv
from scripts import model_store as ms
from scripts import visualization as V

PAGES = ["Ringkasan", "Eksplorasi Data", "Eksplorasi Jarak", "Klasterisasi", "Rekomendasi Metode", "Metodologi"]
NAV_SVG = ["grid", "table", "trend", "network", "trophy", "info"]
ICONS = ["space_dashboard", "query_stats", "timeline", "bubble_chart", "workspace_premium", "menu_book"]
REP_LABEL = ix.REPRESENTATIONS
ALGO_LABEL = ix.ALGORITHMS
Y_TITLE = {"original": "Penjualan asli", "minmax": "Nilai Min-Max", "zscore": "Nilai z-score"}


# =======================================================================================
# Data
# =======================================================================================
@st.cache_resource(show_spinner="Memuat hasil model…")
def load_bundle(mtime: float) -> dict:
    """Memuat model_results.json + data turunan (di-cache; ``mtime`` memaksa muat ulang bila file berubah)."""
    res = ms.load_results(config.RESULTS_FILE)
    W = np.asarray(res["W"], dtype=float)
    codes = np.asarray(res["codes"]).astype(str)
    sales = pd.DataFrame(W, columns=config.WEEK_COLUMNS[: W.shape[1]]).assign(Product_Code=codes)
    from scripts.data import build_product_statistics, build_weekly_statistics
    return {
        "res": res, "W": W, "codes": codes,
        "X": {rep: ix.represent(W, rep) for rep in res["spaces"]},
        "stats": build_product_statistics(sales, "Product_Code", config.WEEK_COLUMNS[: W.shape[1]]),
        "weekly": build_weekly_statistics(sales, config.WEEK_COLUMNS[: W.shape[1]]),
        "table": ms.metrics_table(res),
    }


def bundle() -> dict | None:
    if not config.RESULTS_FILE.is_file():
        return None
    return load_bundle(config.RESULTS_FILE.stat().st_mtime)


def result(b: dict, key: str):
    rep, algo, _, extra = key.split("|")
    return ms.run_to_result(b["res"]["runs"][key], b["res"]["spaces"][rep], linkage=extra if algo == "agglomerative" else None)


def key_label(key: str) -> str:
    rep, algo, k, extra = key.split("|")
    tail = f"eps p{extra}" if algo == "dbscan" else f"K = {k}" + (f" · {extra}" if extra else "")
    return f"{ALGO_LABEL[algo].split(' (')[0]} · {REP_LABEL[rep].split(' (')[0]} · {tail}"


def active_key(b: dict) -> str:
    key = st.session_state.get("active_key")
    if key not in b["res"]["runs"]:
        best = b["res"]["best"]
        key = best.get(config.DEFAULT_REPRESENTATION) or next(v for v in best.values() if v) or next(iter(b["res"]["runs"]))
        st.session_state["active_key"] = key
    return key


# =======================================================================================
# Komponen tampilan
# =======================================================================================
def apply_css() -> None:
    css = config.PROJECT_ROOT / "assets" / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _glow(fig) -> None:
    """Efek neon: garis utama diberi bayangan lebar-transparan. Murni tampilan; data & hover tidak berubah."""
    try:
        import plotly.graph_objects as go
        traces = list(fig.data)
        if len(traces) > 14:
            return
        extra = []
        for t in traces:
            if getattr(t, "type", "") != "scatter" or "lines" not in (t.mode or "") or getattr(t, "stackgroup", None):
                continue
            color, width, dash = t.line.color, t.line.width or 2, t.line.dash
            if not isinstance(color, str) or width < 2 or dash not in (None, "solid") or t.x is None or len(t.x) > 400:
                continue
            extra.append(go.Scatter(x=t.x, y=t.y, mode="lines", line=dict(color=color, width=width * 4, shape=t.line.shape), opacity=0.16,
                                    hoverinfo="skip", showlegend=False, xaxis=t.xaxis, yaxis=t.yaxis))
        if extra:
            fig.add_traces(extra)
    except Exception:
        pass


def plot(fig, key: str) -> None:
    fig.update_layout(title=None)
    _glow(fig)
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "displayModeBar": "hover", "modeBarButtonsToRemove": ["select2d", "lasso2d"]})


_SVG = {
    "box": '<path d="M21 8l-9-5-9 5 9 5 9-5z"/><path d="M3 8v8l9 5 9-5V8"/><path d="M12 13v8"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
    "bars": '<path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20H2"/>',
    "network": '<circle cx="12" cy="5" r="2.2"/><circle cx="5" cy="18" r="2.2"/><circle cx="19" cy="18" r="2.2"/><circle cx="12" cy="12" r="2"/><path d="M12 7.2V10M10.4 13.2l-3.9 3.2M13.6 13.2l3.9 3.2"/>',
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "trend_down": '<path d="M3 7l6 6 4-4 8 8"/><path d="M15 17h6v-6"/>',
    "cube": '<path d="M12 2l9 5v10l-9 5-9-5V7z"/><path d="M12 22V12M21 7l-9 5-9-5"/>',
    "filter": '<path d="M3 4h18l-7 9v6l-4 2v-8z"/>',
    "trophy": '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 01-10 0z"/><path d="M17 5h3a3 3 0 01-3 4M7 5H4a3 3 0 003 4"/>',
    "wave": '<path d="M2 12c2.5-5 5-5 7.5 0s5 5 7.5 0 3.5-3 5-2"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1"/>',
    "scale": '<path d="M12 3v18M5 21h14M5 7h14"/><path d="M5 7l-3 7a3 3 0 006 0zM19 7l-3 7a3 3 0 006 0z"/>',
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    "zero": '<circle cx="12" cy="12" r="8"/><path d="M6.5 17.5l11-11"/>',
    "sliders": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "table": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 4v16"/>',
    "map": '<circle cx="7" cy="8" r="2.5"/><circle cx="16" cy="7" r="2"/><circle cx="12" cy="16" r="3"/><path d="M3 21h18"/>',
    "pulse": '<path d="M3 12h4l3-7 4 14 3-7h4"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 8h0M11 12h1v5h1"/>',
}


def icon(name: str, size: int = 20) -> str:
    """Ikon garis sederhana (SVG inline, warna mengikuti currentColor)."""
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round">{_SVG.get(name, _SVG["info"])}</svg>')


def idn(value, digits: int = 0) -> str:
    """Format angka Indonesia: titik ribuan, koma desimal (375.287 · 66,6)."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def kpis(cards: list[tuple]) -> None:
    """cards = (label, nilai, keterangan[, ikon[, warna aksen]])."""
    for col, card in zip(st.columns(len(cards)), cards):
        label, value, detail = card[:3]
        ic = card[3] if len(card) > 3 else "pulse"
        accent = card[4] if len(card) > 4 else "var(--cyan)"
        col.markdown(f'<div class="kpi" style="--accent:{accent}"><div class="ic">{icon(ic)}</div><div><div class="kpi-label">{label}</div>'
                     f'<div class="kpi-value">{value}</div><div class="kpi-detail">{detail}</div></div></div>', unsafe_allow_html=True)


def section(title: str, sub: str = "", ic: str = "pulse") -> None:
    """Judul section konsisten: ikon cyan + judul + keterangan singkat."""
    st.markdown(f'<div class="section-title">{icon(ic)}<span>{title}</span></div>' + (f'<div class="section-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def best_cards(b: dict, keys: dict) -> None:
    """Tiga kartu model terbaik (Data asli · Min-Max · Scale) yang mudah dibandingkan."""
    style = {"original": ("var(--emerald)", "bars"), "minmax": ("var(--gold)", "wave"), "zscore": ("var(--cyan)", "cube")}
    cols = st.columns(len(keys))
    for col, (rep, key) in zip(cols, keys.items()):
        accent, ic = style.get(rep, ("var(--cyan)", "target"))
        m = b["res"]["runs"][key]["metrics"]
        col.markdown(
            f'<div class="best" style="--accent:{accent}"><div class="bg">{icon(ic, 44)}</div>'
            f'<div class="t">{icon("trophy", 16)} Terbaik · {REP_LABEL[rep]}</div><div class="m">{key_label(key)}</div>'
            f'<div class="s"><span>Silhouette <b>{idn(m["Silhouette"], 3)}</b></span><span>Balance <b>{idn(m["Balance"], 3)}</b></span>'
            f'<span>η² nol <b>{idn(m["Eta2_nol"], 3)}</b></span></div></div>', unsafe_allow_html=True)


def heading(number: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{number} · Ruang Sales</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="page-sub">{subtitle}</p>', unsafe_allow_html=True)


def insight(title: str, color: str, body: str) -> str:
    return f'<div class="insight" style="border-left-color:{color}"><div class="t">{title}</div>{body}</div>'


def fmt(v, f="{:.3f}") -> str:
    """Angka metrik dengan format Indonesia (3 desimal bawaan)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f.format(v).replace(",", "_").replace(".", ",").replace("_", ".")


def quality(s) -> str:
    if s is None or not np.isfinite(s):
        return "Tidak terdefinisi"
    return "Struktur kuat" if s > .7 else "Struktur wajar" if s > .5 else "Struktur lemah" if s > .25 else "Tanpa struktur berarti"


def sidebar(page_objects: dict, current: str) -> None:
    st.markdown(f'<div class="brand"><div class="brand-icon">{icon("layers", 22)}</div><div><div class="brand-name">Ruang Sales</div>'
                '<div class="brand-sub">Sales Analytics &amp; Time Series Clustering</div></div></div>', unsafe_allow_html=True)
    for i, (name, ic) in enumerate(zip(PAGES, ICONS), 1):
        label = f"{i:02d}  {name}"
        if name == current:
            st.markdown(f'<div class="nav-active">{icon(NAV_SVG[i - 1], 20)}{label}</div>', unsafe_allow_html=True)
        else:
            st.page_link(page_objects[name], label=label, icon=f":material/{ic}:")
    b = bundle()
    if b is None:
        return
    key = active_key(b)
    rep, algo, k, extra = key.split("|")
    m = b["res"]["runs"][key]["metrics"]
    rows = [("Representasi", REP_LABEL[rep].split(" (")[0]), ("Cluster", m["Cluster"]), ("Silhouette", fmt(m["Silhouette"])),
            ("Balance", fmt(m["Balance"])), ("η² nol", fmt(m.get("Eta2_nol")))]
    body = "".join(f'<div class="kv"><span>{a}</span><b>{v}</b></div>' for a, v in rows)
    best = '<span class="pill ok">terbaik</span>' if key in b["res"]["best"].values() else ""
    st.markdown(f'<div class="side-note"><div class="side-head">{icon("sliders", 18)}<span>Model Aktif</span>{best}</div>'
                f'<div class="side-model">{ALGO_LABEL[algo].split(" (")[0]}{" · eps p" + extra if algo == "dbscan" else ""}</div>{body}</div>', unsafe_allow_html=True)
    meta = b["res"]["metadata"]
    st.markdown(f'<div class="side-foot">{meta["n_products"]} produk · {meta["n_weeks"]} minggu<br>{len(b["res"]["runs"])} model (Colab) · FastDTW r={meta["distance"]["radius"]}</div>', unsafe_allow_html=True)


def upload_page() -> None:
    """Ditampilkan bila folder hasil_model belum ada: unggah hasil_model.zip dari Colab."""
    heading("00", "Hasil model belum ada.", "Jalankan notebook Colab <b>01_bangun_model</b>, lalu unggah <code>hasil_model.zip</code> di sini.")
    up = st.file_uploader("hasil_model.zip", type="zip")
    if up is not None:
        with zipfile.ZipFile(io.BytesIO(up.getvalue())) as z:
            names = [n for n in z.namelist() if n.startswith("hasil_model/")]
            if "hasil_model/model_results.json" not in names:
                st.error("Zip tidak berisi hasil_model/model_results.json. Pastikan file berasal dari notebook 01_bangun_model.")
                return
            z.extractall(config.PROJECT_ROOT, members=names)
        st.success("Hasil model tersimpan. Memuat dashboard…")
        st.rerun()


def need_bundle() -> dict | None:
    b = bundle()
    if b is None:
        upload_page()
    return b


# =======================================================================================
# 01 · Ringkasan
# =======================================================================================
def page_overview() -> None:
    b = need_bundle()
    if b is None:
        return
    W, weekly, stats, res = b["W"], b["weekly"], b["stats"], b["res"]
    totals = weekly["Total_Penjualan"].to_numpy()
    sorted_totals = np.sort(W.sum(axis=1))[::-1]
    top20 = 100 * sorted_totals[: round(.2 * len(sorted_totals))].sum() / sorted_totals.sum()
    half = 100 * (totals[26:].sum() - totals[:26].sum()) / totals[:26].sum()
    stats_html = "".join(
        f'<div class="hero-stat"><div class="ic{" gold" if gold else ""}">{icon(ic, 24)}</div><div><div class="v">{v}</div><div class="l">{l}</div></div></div>'
        for ic, v, l, gold in [("box", idn(len(W)), "Produk", True), ("layers", idn(W.sum()), "Unit terjual", False),
                               ("bars", f"{idn(top20, 1)}%", "Volume dari 20% produk teratas", True), ("network", idn(len(res["runs"])), "Model dibangun di Colab", False)])
    st.markdown('<div class="hero"><div><div class="eyebrow">Sales Transactions · W0–W51</div>'
                '<h1>Penjualan 52 Minggu,<br>dalam <em>Satu Pandangan</em>.</h1>'
                '<p>Analisis pola penjualan mingguan dengan klasterisasi time series berbasis FastDTW pada tiga representasi data '
                '(Data asli, Min-Max, Scale) dan enam algoritma.</p></div>'
                f'<div class="hero-stats">{stats_html}</div></div>', unsafe_allow_html=True)
    kpis([("Rata-rata / minggu", idn(totals.mean()), f"Puncak W{totals.argmax()}: {idn(totals.max())}", "calendar"),
          ("Perubahan paruh 2 vs 1", f"{'+' if half > 0 else '−'}{idn(abs(half), 1)}%", "W26–W51 dibanding W0–W25", "trend_down" if half < 0 else "trend",
           "var(--neg)" if half < 0 else "var(--emerald)"),
          ("Produk aktif", f"{idn(weekly['Persen_Produk_Aktif'].mean(), 1)}%", "rata-rata per minggu", "cube", "var(--gold)"),
          ("Sel bernilai nol", f"{idn(100 * (W == 0).mean(), 1)}%", "dari 811 produk × 52 minggu", "filter", "var(--ink-2)")])
    keys = {rep: k for rep, k in res["best"].items() if k}
    if keys:
        st.write("")
        best_cards(b, keys)
    st.write("")
    left, right = st.columns([1.55, 1], gap="medium")
    with left, st.container(border=True):
        section("Perjalanan penjualan mingguan", "Total semua produk (cyan) dan rata-rata bergerak 4 minggu (kuning). W0–W51 = urutan minggu, bukan tanggal kalender.", "trend")
        plot(V.create_weekly_sales_figure(weekly, config), "ov_weekly")
        st.caption(f"Penjualan turun {idn(abs(half), 1)}% pada paruh kedua; penurunan paling tajam setelah puncak W{totals.argmax()}." if half < 0
                   else f"Penjualan naik {idn(half, 1)}% pada paruh kedua.")
    with right, st.container(border=True):
        section("Konsentrasi volume (Pareto)", "Produk diurutkan dari volume terbesar; garis titik = distribusi merata.", "bars")
        plot(V.create_pareto_figure(stats, config), "ov_pareto")
        st.caption(f"20% produk teratas menyumbang {idn(top20, 1)}% volume.")
    a, c = st.columns([1.55, 1], gap="medium")
    with a, st.container(border=True):
        section("Hasil klasterisasi", "Ringkasan model terbaik pada tiga representasi data.", "layers")
        rows = []
        for rep, key in keys.items():
            m = res["runs"][key]["metrics"]
            rows.append({"Representasi": REP_LABEL[rep], "Model terbaik": key_label(key).split(" · ")[0] + " · " + key_label(key).split(" · ")[-1],
                         "Silhouette": m["Silhouette"], "DBI": m["DBI_MDS"], "CHI": m["CHI_MDS"], "Balance": m["Balance"], "Cluster": m["Cluster"],
                         "η² nol": m["Eta2_nol"], "Stabilitas": res["metadata"].get("stability", {}).get(rep, {}).get("status", "–")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c2: st.column_config.NumberColumn(format="%.3f") for c2 in ["Silhouette", "DBI", "Balance", "η² nol"]} | {"CHI": st.column_config.NumberColumn(format="%.0f")})
        st.caption("Silhouette antar-representasi tidak sepenuhnya setara (ruang jarak berbeda); baca bersama Balance dan η² minggu nol.")
    with c, st.container(border=True):
        section("Kesamaan pembagian produk (ARI)", "1 = identik, 0 = acak. Konsistensi hasil antar-representasi.", "grid")
        if len(keys) >= 2:
            names = [REP_LABEL[r].split(" (")[0] for r in keys]
            ari = pd.DataFrame([[adjusted_rand_score(res["runs"][keys[x]]["labels"], res["runs"][keys[y]]["labels"]) for y in keys] for x in keys], index=names, columns=names)
            fig = V.create_similarity_heatmap(ari, "ARI", config)
            fig.update_layout(height=330)
            plot(fig, "ov_ari")
    a, c = st.columns(2, gap="medium")
    key = active_key(b)
    labels = np.asarray(res["runs"][key]["labels"])
    with a, st.container(border=True):
        section("10 produk dengan volume tertinggi", f"Warna = cluster model aktif ({key_label(key)}).", "trophy")
        plot(V.create_top_products_figure(stats, 10, config, dict(zip(b["codes"], labels))), "ov_top")
    with c, st.container(border=True):
        section("Produk aktif & minggu tanpa penjualan", "Persentase produk terjual per minggu, lalu sebaran jumlah minggu nol per produk.", "zero")
        plot(V.create_active_products_figure(weekly, config), "ov_active")
        plot(V.create_zero_weeks_figure(stats, config), "ov_zero")


# =======================================================================================
# 02 · Eksplorasi data
# =======================================================================================
def page_exploration() -> None:
    b = need_bundle()
    if b is None:
        return
    heading("02", "Kenali data sebelum mengelompokkan.", "Detail per produk pada tiga representasi, hubungan aktivitas–volume, dan peta panas seluruh produk.")
    key = active_key(b)
    labels = np.asarray(b["res"]["runs"][key]["labels"])
    codes = b["codes"].tolist()
    with st.container(border=True):
        h1, h2 = st.columns([3, 1.4])
        h1.subheader(f"Peta panas {len(codes)} produk × 52 minggu")
        h1.caption(f"Dikelompokkan menurut model aktif ({key_label(key)}), lalu volume terbesar. Putih = 0 unit.")
        mode = h2.segmented_control("Warna", ["Min-Max", "Scale (z-score)", "Log data asli"], default="Min-Max", key="ex_heat") or "Min-Max"
        heat = {"Min-Max": "minmax", "Scale (z-score)": "zscore"}.get(mode, "log")
        plot(V.create_sales_heatmap_figure(b["W"], b["X"]["zscore" if heat == "zscore" else "minmax"], labels, b["codes"], heat, config), "ex_heatmap")
    left, right = st.columns([1.55, 1], gap="medium")
    with left, st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1.3])
        prod = c1.selectbox("Produk", codes, key="ex_prod")
        cmp = c2.selectbox("Bandingkan", ["—", *codes], key="ex_cmp")
        scale_name = c3.segmented_control("Skala", ["Data asli", "Min-Max", "Scale"], default="Data asli", key="ex_scale") or "Data asli"
        scale = {"Data asli": "raw", "Min-Max": "minmax", "Scale": "zscore"}[scale_name]
        figs = V.create_exploration_figures(b["stats"], b["W"], prod, config, labels=labels, compare_product=None if cmp == "—" else cmp,
                                            scale=scale, X_scaled=b["X"].get(scale) if scale != "raw" else None)
        section(f"Deret mingguan · {prod}", "Garis titik = median cluster produk ini sebagai konteks.", "trend")
        plot(figs["product_detail"], "ex_series")
    with right, st.container(border=True):
        section("Label di setiap representasi", "Cluster produk ini pada model terbaik tiap representasi.", "layers")
        i = codes.index(prod)
        rows = []
        for rep, bk in b["res"]["best"].items():
            if bk:
                run = b["res"]["runs"][bk]
                rows.append({"Representasi": REP_LABEL[rep], "Model": key_label(bk).split(" · ")[0], "Cluster": V.cluster_name(run["labels"][i]),
                             "Silhouette": run["silhouette"][i]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={"Silhouette": st.column_config.NumberColumn(format="%.3f")})
        row = b["stats"].iloc[i]
        st.metric("Total volume", f"{row['Total volume']:,.0f}")
        st.metric("Minggu nol", f"{int(row['Minggu nol'])} / 52")
    a, c = st.columns(2, gap="medium")
    with a, st.container(border=True):
        section("Aktivitas vs volume", "Sumbu Y skala log; warna = cluster model aktif.", "map")
        plot(figs["activity_vs_volume"], "ex_scatter")
    with c, st.container(border=True):
        section("Distribusi total volume", "Bin logaritmik: distribusi sangat menceng ke kanan.", "bars")
        plot(figs["sales_distribution"], "ex_hist")


# =======================================================================================
# 03 · Eksplorasi jarak
# =======================================================================================
def _fastdtw(a: np.ndarray, b: np.ndarray, radius: int):
    """FastDTW (numba) dengan jalur; jika numba tidak ada, CDTW pita = radius sebagai aproksimasi."""
    try:
        from scripts.distance import fastdtw_numba
        d, pi, pj = fastdtw_numba(np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64), int(radius), True)
        return float(d), np.c_[pi, pj], "FastDTW (numba)"
    except Exception:
        d, path, _ = ix.dtw_pair(a, b, int(radius))
        return d, path, f"CDTW pita {radius} (numba tidak tersedia)"


def page_distance() -> None:
    b = need_bundle()
    if b is None:
        return
    heading("03", "Seberapa mirip dua produk?", "Euclidean vs DTW vs FastDTW pada representasi pilihan, lengkap dengan jalur penyejajaran minggu.")
    codes = b["codes"].tolist()
    radius = b["res"]["metadata"]["distance"]["radius"]
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        rep = c1.selectbox("Representasi", list(b["X"]), format_func=REP_LABEL.get, key="dj_rep")
        pa = c2.selectbox("Produk A", codes, index=0, key="dj_a")
        pb = c3.selectbox("Produk B", codes, index=1, key="dj_b")
    X = b["X"][rep]
    ia, ib = codes.index(pa), codes.index(pb)
    eu = float(np.sqrt(((X[ia] - X[ib]) ** 2).sum()))
    d_dtw, path_dtw, cost = ix.dtw_pair(X[ia], X[ib], None)
    d_fast, path_fast, engine = _fastdtw(X[ia], X[ib], radius)
    kpis([("Euclidean", fmt(eu), "minggu dipasangkan 1-ke-1", "grid", "var(--ink-2)"), ("DTW penuh", fmt(d_dtw), "jalur optimal eksak", "pulse"),
          (f"FastDTW r = {radius}", fmt(d_fast), engine, "wave", "var(--gold)"), ("Pasangan bergeser", f"{int((path_fast[:, 0] != path_fast[:, 1]).sum())} / {len(path_fast)}", "langkah jalur di luar diagonal", "trend", "var(--emerald)")])
    left, right = st.columns([1.3, 1], gap="medium")
    with left, st.container(border=True):
        section("Week alignment · FastDTW", "Garis abu-abu = minggu yang disejajarkan; miring = pergeseran waktu.", "wave")
        plot(iv.dtw_alignment_figure(X[ia], X[ib], path_fast, pa, pb, Y_TITLE[rep], config), "dj_align")
    with right, st.container(border=True):
        section("Matriks biaya & jalur", "Warna = biaya kumulatif DTW penuh; merah = jalur FastDTW.", "grid")
        plot(iv.dtw_cost_figure(cost, path_fast, None, pa, pb, config), "dj_cost")
    with st.container(border=True):
        section(f"10 produk paling mirip dengan {pa}", "FastDTW ke seluruh produk pada representasi terpilih.", "target")
        dist = np.array([_fastdtw(X[ia], X[j], radius)[0] for j in range(len(X))])
        frame = pd.DataFrame({"Produk": codes, "FastDTW": dist, "Euclidean": np.sqrt(((X - X[ia]) ** 2).sum(axis=1)),
                              "Total volume": b["W"].sum(axis=1), "Pola": [list(map(float, r)) for r in X]}).drop(index=ia).nsmallest(10, "FastDTW")
        st.dataframe(frame, hide_index=True, width="stretch", column_config={"FastDTW": st.column_config.NumberColumn(format="%.3f"),
                     "Euclidean": st.column_config.NumberColumn(format="%.3f"), "Pola": st.column_config.LineChartColumn(width="medium")})


# =======================================================================================
# 04 · Klasterisasi (pilih → Tampilkan)
# =======================================================================================
def _apply_preset() -> None:
    b = bundle()
    choice = st.session_state.get("cl_preset")
    if not b or not choice or choice == "Kustom":
        return
    rep, algo, k, extra = b["res"]["best"][choice].split("|")
    st.session_state.update(cl_rep=rep, cl_algo=algo)
    if algo == "dbscan":
        st.session_state["cl_eps"] = int(extra)
    else:
        st.session_state["cl_k"] = int(k)
    if algo == "agglomerative":
        st.session_state["cl_link"] = extra


def page_clustering() -> None:
    b = need_bundle()
    if b is None:
        return
    heading("04", "Hasil klasterisasi.", "Pilih representasi, algoritma, dan parameter, lalu tekan <b>Tampilkan hasil</b>. Semua model sudah dibangun di Colab dengan FastDTW.")
    cfg = b["res"]["metadata"]["config"]
    best = b["res"]["best"]
    with st.container(border=True):
        section("Kontrol parameter", "Preset mengisi pilihan dengan model terbaik; ubah sesuai kebutuhan lalu tekan Tampilkan hasil.", "sliders")
        st.selectbox("Preset", ["Kustom", *[r for r, k in best.items() if k]], key="cl_preset", on_change=_apply_preset,
                     format_func=lambda r: r if r == "Kustom" else f"Model terbaik · {REP_LABEL[r]} ({key_label(best[r]).split(' · ', 2)[0]})")
        c1, c2, c3, c4 = st.columns(4)
        rep = c1.selectbox("Representasi data", list(b["res"]["spaces"]), format_func=REP_LABEL.get, key="cl_rep",
                           help="Data asli: volume ikut menentukan. Min-Max: bentuk 0–1. Scale: bentuk relatif terhadap rata-rata & simpangan produk.")
        algo = c2.selectbox("Algoritma", ms.ALGORITHMS, format_func=ALGO_LABEL.get, key="cl_algo")
        if algo == "dbscan":
            extra = c3.selectbox("eps (persentil jarak tetangga)", cfg["DBSCAN_EPS_PERCENTILES"], index=len(cfg["DBSCAN_EPS_PERCENTILES"]) // 2,
                                 format_func=lambda p: f"persentil {p}", key="cl_eps")
            key = ms.run_key(rep, algo, 0, extra)
        else:
            k = c3.select_slider("Jumlah cluster K", cfg["K_GRID"], key="cl_k")
            extra = c4.selectbox("Linkage", cfg["LINKAGES"], key="cl_link") if algo == "agglomerative" else None
            key = ms.run_key(rep, algo, k, extra)
        st.caption({"kmeans": "Centroid = barycenter DBA (rata-rata yang menghormati pergeseran waktu).",
                    "kmedoids": "Centroid = medoid (produk nyata), PAM BUILD + SWAP.",
                    "agglomerative": "Tidak punya pusat bawaan; medoid tiap cluster + dendrogram.",
                    "dbscan": f"min_samples = {cfg['DBSCAN_MIN_SAMPLES']}; produk di wilayah jarang = noise.",
                    "fcm": f"Fuzzy m = {cfg['FUZZINESS']}; centroid DBA berbobot keanggotaan.",
                    "fcmdd": f"Fuzzy m = {cfg['FUZZINESS']}; centroid = medoid berbobot keanggotaan."}[algo])
    b1, b2 = st.columns([1, 3])
    if b1.button("Tampilkan hasil", type="primary", icon=":material/play_arrow:", width="stretch"):
        st.session_state["shown_key"] = key
        st.session_state["active_key"] = key
    shown = st.session_state.get("shown_key")
    if shown and shown != key:
        b2.warning("Pilihan berubah sejak hasil terakhir. Tekan **Tampilkan hasil** untuk memperbarui.", icon=":material/sync_problem:")
    if not shown:
        st.markdown('<div class="callout"><b>Belum ada hasil.</b> Pilih skenario di atas lalu tekan <b>Tampilkan hasil</b>. '
                    'Pilih <i>Preset</i> untuk langsung melihat model terbaik tiap representasi.</div>', unsafe_allow_html=True)
        return
    render_result(b, shown)


def render_result(b: dict, key: str) -> None:
    rep, algo, k, extra = key.split("|")
    r = result(b, key)
    space = b["res"]["spaces"][rep]
    X, W, codes = b["X"][rep], b["W"], b["codes"]
    m = r.metrics
    kpis([("Silhouette", fmt(m["Silhouette"]), f"{quality(m['Silhouette'])} · pemisahan antar-cluster (↑)", "target", "var(--cyan)"),
          ("DBI (MDS)", fmt(m["DBI_MDS"]), "Davies–Bouldin: makin kecil makin terpisah (↓)", "scale", "var(--gold)"),
          ("CHI (MDS)", idn(m["CHI_MDS"]), "Calinski–Harabasz: makin besar makin padat (↑)", "bars", "var(--emerald)"),
          ("Balance", fmt(m["Balance"]), "1 = ukuran cluster seimbang", "layers", "var(--cyan)"),
          ("Jumlah cluster", str(int(m["Cluster"])), f"coverage {idn(100 * m['Coverage'], 1)}%" + (f" · {int(m['Noise'])} noise" if m["Noise"] else ""), "network", "var(--gold)"),
          ("η² minggu nol", fmt(m["Eta2_nol"]), "tinggi = cluster dipisah frekuensi nol", "zero", "var(--neg)")])
    is_best = key in b["res"]["best"].values()
    st.caption(f"{key_label(key)} · FastDTW · pusat: {ix.CENTER_KIND[r.center_kind]}" + (" · ★ model terbaik representasi ini" if is_best else ""))
    for note in r.notes:
        st.caption(f"• {note}")
    if m["Cluster"] < 2:
        st.warning("Hanya terbentuk satu cluster" + (" padat; sisanya noise. Pilih persentil eps lain." if algo == "dbscan" else "."), icon=":material/info:")
    tabs = st.tabs(["Peta cluster & centroid", "Pola centroid", "Kualitas & diagnostik", "Profil & interpretasi", "Peta panas anggota"])
    with tabs[0]:
        c1, c2, _ = st.columns([1.2, 1, 2])
        shape = c1.segmented_control("Batas cluster", ["hull", "ellipse", "none"], default="hull", key="fv_shape",
                                     format_func={"hull": "Convex hull", "ellipse": "Elips 95%", "none": "Tanpa"}.get) or "hull"
        spider = c2.toggle("Garis ke centroid", key="fv_spider")
        st.caption("Gaya `fviz_cluster`: koordinat MDS klasik dari matriks FastDTW. ◆ = centroid (DBA diproyeksikan formula Gower), ★ = medoid / produk representatif.")
        xy = np.array(space["xy"], dtype=float)
        plot(iv.fviz_cluster_figure(xy, np.array(space["explained"], dtype=float), r.labels, codes, r.center_xy, r.cluster_ids, r.medoids,
                                    r.center_kind, shape, spider, config, "Dim"), "fv_map")
        a, c = st.columns([1.6, 1], gap="medium")
        with a, st.container(border=True):
            section("Semua centroid", "Bentuk pola tiap cluster dalam satu grafik.", "trend")
            if len(r.cluster_ids):
                plot(iv.centroid_overlay_figure(r.centers, r.cluster_ids, r.center_kind, Y_TITLE[rep], config), "fv_overlay")
        with c, st.container(border=True):
            section("Jarak antar-centroid", "FastDTW/DTW; makin jauh makin terpisah.", "grid")
            if len(r.cluster_ids) >= 2:
                plot(iv.center_distance_figure(r.center_dm, r.cluster_ids, config), "fv_cdm")
    with tabs[1]:
        view = st.segmented_control("Skala", ["rep", "raw"], default="rep", key="cs_view",
                                    format_func={"rep": f"Representasi ({REP_LABEL[rep]})", "raw": "Penjualan asli"}.get) or "rep"
        if len(r.cluster_ids):
            if view == "raw":
                centers = np.stack([W[r.labels == c].mean(axis=0) for c in r.cluster_ids])
                plot(iv.centroid_series_figure(W, r.labels, centers, r.cluster_ids, r.medoids, codes, "mean", "Penjualan asli", config), "cs_raw")
            else:
                plot(iv.centroid_series_figure(X, r.labels, r.centers, r.cluster_ids, r.medoids, codes, r.center_kind, Y_TITLE[rep], config), "cs_rep")
            for col, c, med in zip(st.columns(len(r.cluster_ids)), r.cluster_ids, r.medoids):
                mask = r.labels == c
                col.markdown(f'<div class="persona" style="min-height:0;text-align:center"><div class="bar" style="background:{V.cluster_color(c, config)}"></div>'
                             f'<div class="kpi-label">Produk representatif {V.cluster_name(c)}</div><h4 style="font-size:24px">{codes[med]}</h4>'
                             f'<span class="chip">{mask.sum()} produk</span> <span class="chip">Total {W[mask].sum():,.0f}</span></div>', unsafe_allow_html=True)
    with tabs[2]:
        a, c = st.columns(2, gap="medium")
        with a, st.container(border=True):
            section("Plot silhouette", "Gaya fviz_silhouette; batang < 0 = mungkin salah cluster.", "target")
            if np.isfinite(m["Silhouette"]):
                plot(iv.fviz_silhouette_figure(r.silhouette, r.labels, codes, config), "q_sil")
                st.caption(f"{100 * np.nanmean(r.silhouette[r.labels != -1] < 0):.1f}% produk bernilai negatif (lebih dekat ke cluster lain).")
        with c, st.container(border=True):
            section("Jarak anggota ke pusat", "Ekor panjang = anggota pinggiran.", "scale")
            plot(iv.distance_to_center_figure(r.dist_to_center, r.labels, codes, config), "q_d2c")
        if algo == "agglomerative":
            with st.container(border=True):
                section(f"Dendrogram · linkage {extra}", "Cabang diwarnai per cluster; garis merah = posisi potong.", "network")
                plot(iv.fviz_dend_figure(r.linkage, r.labels, int(k), config), "q_dend")
        if algo == "dbscan":
            with st.container(border=True):
                section("kNN-distance plot", "eps ideal di lutut kurva.", "trend")
                plot(iv.knn_distance_figure(np.array(space["kdist"], dtype=float), r.eps, b["res"]["metadata"]["config"]["DBSCAN_MIN_SAMPLES"], config), "q_knn")
        if r.membership is not None:
            with st.container(border=True):
                section("Keanggotaan fuzzy", "", "layers")
                st.caption(f"FPC {fmt(m.get('FPC'))} · entropi partisi {fmt(m.get('Entropi_partisi'))}. Mendekati 0,5 = produk di perbatasan.")
                plot(iv.membership_stack_figure(r.membership, r.labels, codes, config), "q_mem")
        if algo != "dbscan":
            with st.container(border=True):
                section("Berapa K yang tepat?", "Metode siku + rata-rata silhouette dari model K = 2…6 yang sudah dibangun.", "bars")
                rows = [{"k": kk, **b["res"]["runs"][ms.run_key(rep, algo, kk, extra)]["metrics"]} for kk in b["res"]["metadata"]["config"]["K_GRID"]]
                rows = [{**row, **{c: (np.nan if v is None else v) for c, v in row.items()}} for row in rows]
                plot(iv.fviz_nbclust_figure(rows, int(k), config), "q_nb")
    with tabs[3]:
        prof = pd.DataFrame(ix.cluster_profile_rows(r, W, codes))
        prof.insert(0, "Nama", prof["Cluster"].map(V.cluster_name))
        section("Ringkasan anggota dan produk representatif", "", "table")
        st.dataframe(prof[["Nama", "Persona", "Jumlah_Produk", "Proporsi_%", "Total_Penjualan", "Porsi_Volume_%", "Produk_Representatif", "Rata2_Jarak_ke_Pusat"]],
                     hide_index=True, width="stretch", column_config={"Porsi_Volume_%": st.column_config.ProgressColumn("Porsi volume", format="%.1f%%", min_value=0, max_value=100),
                                                                       "Proporsi_%": st.column_config.NumberColumn(format="%.1f"), "Rata2_Jarak_ke_Pusat": st.column_config.NumberColumn(format="%.3f")})
        section("Profilisasi deskriptif", "Dihitung dari penjualan asli.", "pulse")
        st.dataframe(prof[["Nama", "Rata2_Mingguan", "Median_Minggu_Nol", "CV_Profil", "Perubahan_13mg_%", "Tren"]], hide_index=True, width="stretch",
                     column_config={"Rata2_Mingguan": st.column_config.NumberColumn(format="%.2f"), "CV_Profil": st.column_config.NumberColumn(format="%.3f"),
                                    "Perubahan_13mg_%": st.column_config.NumberColumn("13 mg awal → akhir", format="%+.1f%%")})
        section("Interpretasi dan rekomendasi bisnis", "Deskriptif dari data historis; validasi dengan konteks bisnis.", "info")
        for row in prof.to_dict("records"):
            ch = row["Perubahan_13mg_%"]
            st.markdown(f'<div class="insight" style="min-height:0;margin-bottom:10px;border-left-color:{V.cluster_color(row["Cluster"], config)}">'
                        f'<b>{row["Nama"]} · {row["Tren"]}</b> &nbsp;<span class="chip">{row["Persona"]}</span><br><span class="kpi-detail">{row["Jumlah_Produk"]} produk · '
                        f'13 minggu awal → akhir: {"–" if not np.isfinite(ch) else f"{ch:+.1f}%"} · CV {row["CV_Profil"]:.3f}</span><br>{row["Rekomendasi"]}</div>', unsafe_allow_html=True)
        members = pd.DataFrame({"Product_Code": codes, "Cluster": [V.cluster_name(c) for c in r.labels], "Total": W.sum(axis=1),
                                "Minggu_nol": (W == 0).sum(axis=1), "Silhouette": r.silhouette, "Jarak_ke_pusat": r.dist_to_center})
        if r.membership is not None:
            members["Keanggotaan"] = r.membership.max(axis=1)
        pick = st.selectbox("Lihat anggota", sorted(members["Cluster"].unique()), key="pf_pick")
        st.dataframe(members[members["Cluster"] == pick].sort_values("Total", ascending=False), hide_index=True, width="stretch", height=320,
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Silhouette", "Jarak_ke_pusat", "Keanggotaan"]})
        st.download_button("Unduh label (CSV)", members.to_csv(index=False).encode("utf-8-sig"), f"label_{key.replace('|', '_')}.csv", "text/csv", icon=":material/download:")
    with tabs[4]:
        st.caption("Per cluster, diurutkan dari produk paling dekat ke pusat (atas) ke paling jauh (bawah).")
        plot(iv.cluster_heatmap_figure(X, r.labels, r.dist_to_center, codes, config), "hm")


# =======================================================================================
# 05 · Rekomendasi
# =======================================================================================
def page_recommendation() -> None:
    b = need_bundle()
    if b is None:
        return
    heading("05", "Model mana yang paling cocok?", "Model terbaik tiap representasi dipilih di Colab: lolos guardrail lalu peringkat rata-rata Silhouette↑, DBI↓, CHI↑.")
    res, meta = b["res"], b["res"]["metadata"]
    cfg = meta["config"]
    st.caption(f"Guardrail: minimal {cfg['MIN_CLUSTERS']} cluster · Balance ≥ {cfg['MIN_BALANCE']} · noise ≤ {int(100 * cfg['MAX_NOISE_SHARE'])}% produk.")
    t1, t2, t3 = st.tabs(["Terbaik per representasi", "Peringkat per representasi", "Semua model"])
    with t1:
        best_cards(b, {rep: k for rep, k in res["best"].items() if k})
        st.write("")
        rows = []
        for rep, key in res["best"].items():
            if key:
                m = res["runs"][key]["metrics"]
                rows.append({"Representasi": REP_LABEL[rep], "Model terbaik": key_label(key), **{c: m[c] for c in ["Silhouette", "DBI_MDS", "CHI_MDS", "Balance", "Cluster", "Eta2_nol"]},
                             "Stabilitas seed": meta.get("stability", {}).get(rep, {}).get("status", "–")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Silhouette", "DBI_MDS", "Balance", "Eta2_nol"]} | {"CHI_MDS": st.column_config.NumberColumn(format="%.0f")})
        keys = {rep: k for rep, k in res["best"].items() if k}
        if len(keys) >= 2:
            names = [REP_LABEL[r].split(" (")[0] for r in keys]
            ari = pd.DataFrame([[adjusted_rand_score(res["runs"][keys[a]]["labels"], res["runs"][keys[c]]["labels"]) for c in keys] for a in keys], index=names, columns=names)
            section("Kesamaan pembagian produk (ARI)", "1 = identik, 0 = tidak lebih mirip dari acak. ARI rendah = pilihan representasi benar-benar mengubah segmentasi.", "grid")
            plot(V.create_similarity_heatmap(ari, "ARI", config), "rc_ari")
        if st.button("Buka model terbaik representasi default di Klasterisasi", icon=":material/arrow_forward:"):
            st.session_state["shown_key"] = st.session_state["active_key"] = res["best"].get(config.DEFAULT_REPRESENTATION) or next(iter(keys.values()))
            st.switch_page(st.session_state["_pages"]["Klasterisasi"])
    with t2:
        rep = st.segmented_control("Representasi", list(res["spaces"]), default=config.DEFAULT_REPRESENTATION, format_func=REP_LABEL.get, key="rc_rep") or config.DEFAULT_REPRESENTATION
        table = b["table"][b["table"]["Kunci"].str.startswith(rep + "|")].copy()
        _, ranked = ms.select_best(res["runs"], rep, meta["n_products"], type("C", (), cfg)())
        rank_map = dict(zip(ranked["Kunci"], ranked.get("Skor_Ranking", pd.Series(dtype=float)))) if "Skor_Ranking" in ranked else {}
        table["Skor ranking"] = table["Kunci"].map(rank_map)
        table["Lolos guardrail"] = table["Kunci"].isin(rank_map)
        table = table.sort_values(["Lolos guardrail", "Skor ranking", "Silhouette"], ascending=[False, True, False])
        st.dataframe(table[["Algoritma", "K", "Parameter", "Lolos guardrail", "Skor ranking", "Silhouette", "DBI_MDS", "CHI_MDS", "Balance", "Cluster", "Noise", "Eta2_nol", "Terbaik"]],
                     hide_index=True, width="stretch", height=520,
                     column_config={"Silhouette": st.column_config.ProgressColumn(format="%.3f", min_value=-1, max_value=1),
                                    "Balance": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                                    "Skor ranking": st.column_config.NumberColumn(format="%.2f"), "DBI_MDS": st.column_config.NumberColumn(format="%.3f"),
                                    "CHI_MDS": st.column_config.NumberColumn(format="%.0f"), "Eta2_nol": st.column_config.NumberColumn("η² nol", format="%.3f")})
    with t3:
        st.dataframe(b["table"].drop(columns=["Kunci"]), hide_index=True, width="stretch", height=520)
        st.download_button("Unduh metrik semua model (CSV)", b["table"].to_csv(index=False).encode("utf-8-sig"), "metrik_semua_model.csv", "text/csv", icon=":material/download:")


# =======================================================================================
# 06 · Metodologi
# =======================================================================================
def page_methodology() -> None:
    b = need_bundle()
    if b is None:
        return
    heading("06", "Dari data mentah menjadi pola.", "Alur pembangunan model di Colab dan cara membaca metriknya.")
    meta = b["res"]["metadata"]
    cfg = meta["config"]
    steps = [("01", "Baca & validasi", "CSV Product_Code + W0–W51 (pemisah ; atau , otomatis), 811 produk × 52 minggu."),
             ("02", "Tiga representasi", "Data asli, Min-Max per produk (0–1), Scale/z-score per produk."),
             ("03", f"FastDTW r={meta['distance']['radius']}", "Satu matriks jarak 811 × 811 per representasi (numba)."),
             ("04", "Enam algoritma", f"K-Means DBA, K-Medoids PAM, Agglomerative, DBSCAN, Fuzzy C-Means DBA, Fuzzy C-Medoids · K {cfg['K_GRID']}."),
             ("05", "Evaluasi & pilih", "Silhouette, DBI/CHI pada MDS, Balance, η² minggu nol; guardrail lalu peringkat.")]
    for col, (n, t, d) in zip(st.columns(5), steps):
        col.markdown(f'<div class="persona" style="min-height:170px"><div class="step-n">{n}</div><h4 style="font-size:16px">{t}</h4><div class="meta">{d}</div></div>', unsafe_allow_html=True)
    st.write("")
    a, c = st.columns(2, gap="medium")
    with a, st.container(border=True):
        section("Kamus metrik", "", "info")
        for term, desc in [("Silhouette", "Pemisahan vs kekompakan pada matriks FastDTW; −1…1, lebih tinggi lebih baik (> 0,25 = struktur lemah, > 0,5 = wajar)."),
                           ("DBI & CHI (MDS)", "Dihitung pada embedding MDS 5 dimensi dari matriks FastDTW. DBI ↓, CHI ↑."),
                           ("Balance", "Entropi ukuran cluster ternormalisasi; 1 = seimbang, 0 = satu cluster dominan."),
                           ("η² minggu nol", "Seberapa besar variasi jumlah minggu nol dijelaskan label; tinggi = cluster didorong frekuensi nol, bukan bentuk."),
                           ("ARI", "Kesamaan dua pembagian produk (antar-representasi / antar-seed); bukan akurasi.")]:
            st.markdown(f"**{term}**")
            st.caption(desc)
    with c, st.container(border=True):
        section("Konfigurasi pembangunan model", "", "sliders")
        st.json({"dibangun_utc": meta["created_at_utc"], "durasi_detik": meta["runtime_s"], "jarak": meta["distance"], "konfigurasi": cfg,
                 "model_terbaik": b["res"]["best"], "stabilitas": meta.get("stability"), "versi": meta["versions"]}, expanded=False)
    files = [p for p in sorted(config.RESULTS_DIR.iterdir()) if p.is_file()]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=f"hasil_model/{p.name}")
    st.download_button("Unduh hasil_model.zip", buf.getvalue(), "hasil_model.zip", "application/zip", icon=":material/folder_zip:")
    st.warning("Hasil bersifat deskriptif/eksploratif dari data historis, bukan prediksi. Validasi interpretasi cluster dengan konteks bisnis.", icon=":material/info:")
