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


def plot(fig, key: str) -> None:
    fig.update_layout(title=None)
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "displayModeBar": "hover", "modeBarButtonsToRemove": ["select2d", "lasso2d"]})


def kpis(cards: list[tuple[str, str, str]]) -> None:
    for col, (label, value, detail) in zip(st.columns(len(cards)), cards):
        col.markdown(f'<div class="kpi" style="min-height:118px"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
                     f'<div class="kpi-detail">{detail}</div></div>', unsafe_allow_html=True)


def heading(number: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{number} · Ruang Sales</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="page-sub">{subtitle}</p>', unsafe_allow_html=True)


def insight(title: str, color: str, body: str) -> str:
    return f'<div class="insight" style="border-left-color:{color}"><div class="t">{title}</div>{body}</div>'


def fmt(v, f="{:.3f}") -> str:
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def quality(s) -> str:
    if s is None or not np.isfinite(s):
        return "Tidak terdefinisi"
    return "Struktur kuat" if s > .7 else "Struktur wajar" if s > .5 else "Struktur lemah" if s > .25 else "Tanpa struktur berarti"


def sidebar(page_objects: dict, current: str) -> None:
    st.markdown('<div class="brand"><div class="brand-icon">↗</div><div><div class="brand-name">Ruang Sales</div>'
                '<div class="brand-sub">CERITA DI BALIK ANGKA</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">NAVIGASI</div>', unsafe_allow_html=True)
    for i, (name, icon) in enumerate(zip(PAGES, ICONS), start=1):
        st.button(f"{i:02d}  {name}", key=f"nav_{name}", icon=f":material/{icon}:", width="stretch",
                  type="primary" if name == current else "secondary",
                  on_click=lambda target=name: st.switch_page(page_objects[target]))
    b = bundle()
    if b is None:
        return
    key = active_key(b)
    m = b["res"]["runs"][key]["metrics"]
    st.markdown(f'<div class="side-note"><b>Model aktif</b><br>{key_label(key)}<br>{m["Cluster"]} cluster · Silhouette {fmt(m["Silhouette"])}<br>'
                f'Balance {fmt(m["Balance"])}{" · <span class=ok>terbaik</span>" if key in b["res"]["best"].values() else ""}</div>', unsafe_allow_html=True)
    meta = b["res"]["metadata"]
    st.caption(f"{meta['n_products']} produk · {meta['n_weeks']} minggu · FastDTW radius {meta['distance']['radius']} · model dibangun {meta['created_at_utc'][:10]}")


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
    st.markdown('<div class="hero"><div><div class="eyebrow">Sales Transactions · W0–W51</div><h1>Penjualan 52 minggu, dalam satu pandangan.</h1>'
                '<p>Model dibangun di Colab dengan FastDTW pada tiga representasi data; dashboard ini menampilkan hasilnya tanpa melatih ulang.</p></div>'
                '<div class="hero-stats">' + "".join(f'<div class="hero-stat"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in [
                    (f"{len(W):,}", "produk"), (f"{W.sum():,.0f}", "unit terjual"), (f"{top20:.1f}%", "volume dari 20% produk teratas"),
                    (f"{len(res['runs'])}", "model dibangun di Colab")]) + "</div></div>", unsafe_allow_html=True)
    kpis([("Rata-rata / minggu", f"{totals.mean():,.0f}", f"Puncak W{totals.argmax()}: {totals.max():,.0f}"),
          ("Paruh 2 vs 1", f"{half:+.1f}%", "W26–W51 dibanding W0–W25"),
          ("Produk aktif", f"{weekly['Persen_Produk_Aktif'].mean():.1f}%", "rata-rata per minggu"),
          ("Sel bernilai nol", f"{100 * (W == 0).mean():.1f}%", "produk × minggu")])
    best_cards = []
    for i, (rep, key) in enumerate(res["best"].items()):
        if key:
            m = res["runs"][key]["metrics"]
            best_cards.append(insight(f"Terbaik · {REP_LABEL[rep]}", config.PLOT_PALETTE[i], f"<b>{key_label(key)}</b><br>Silhouette {fmt(m['Silhouette'])} · "
                                      f"Balance {fmt(m['Balance'])} · η² nol {fmt(m['Eta2_nol'])}"))
    if best_cards:
        st.write("")
        for col, card in zip(st.columns(len(best_cards)), best_cards):
            col.markdown(card, unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.55, 1], gap="medium")
    with left, st.container(border=True):
        st.subheader("Perjalanan penjualan mingguan")
        st.caption("Total semua produk + rata-rata bergerak 4 minggu. W0–W51 = urutan minggu, bukan tanggal kalender.")
        plot(V.create_weekly_sales_figure(weekly, config), "ov_weekly")
        st.markdown("**Produk aktif per minggu** · grafik terpisah, bukan sumbu ganda")
        plot(V.create_active_products_figure(weekly, config), "ov_active")
    with right, st.container(border=True):
        st.subheader("Konsentrasi volume (Pareto)")
        plot(V.create_pareto_figure(stats, config), "ov_pareto")
    key = active_key(b)
    labels = np.asarray(res["runs"][key]["labels"])
    a, c = st.columns(2, gap="medium")
    with a, st.container(border=True):
        st.subheader("10 produk dengan volume tertinggi")
        st.caption(f"Warna = cluster model aktif ({key_label(key)}).")
        plot(V.create_top_products_figure(stats, 10, config, dict(zip(b["codes"], labels))), "ov_top")
    with c, st.container(border=True):
        st.subheader("Sebaran minggu tanpa penjualan")
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
        st.subheader(f"Deret mingguan · {prod}")
        plot(figs["product_detail"], "ex_series")
    with right, st.container(border=True):
        st.subheader("Label di setiap representasi")
        st.caption("Cluster produk ini pada model terbaik tiap representasi.")
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
        st.subheader("Aktivitas vs volume")
        plot(figs["activity_vs_volume"], "ex_scatter")
    with c, st.container(border=True):
        st.subheader("Distribusi total volume")
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
    kpis([("Euclidean", f"{eu:.3f}", "minggu 1-ke-1"), ("DTW penuh", f"{d_dtw:.3f}", "jalur optimal eksak"),
          (f"FastDTW r={radius}", f"{d_fast:.3f}", engine), ("Pasangan bergeser", f"{int((path_fast[:, 0] != path_fast[:, 1]).sum())} / {len(path_fast)}", "jalur FastDTW")])
    left, right = st.columns([1.3, 1], gap="medium")
    with left, st.container(border=True):
        st.subheader("Week alignment · FastDTW")
        plot(iv.dtw_alignment_figure(X[ia], X[ib], path_fast, pa, pb, Y_TITLE[rep], config), "dj_align")
    with right, st.container(border=True):
        st.subheader("Matriks biaya & jalur")
        st.caption("Warna = biaya kumulatif DTW penuh; merah = jalur FastDTW.")
        plot(iv.dtw_cost_figure(cost, path_fast, None, pa, pb, config), "dj_cost")
    with st.container(border=True):
        st.subheader(f"10 produk paling mirip dengan {pa}")
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
    with st.expander("Kontrol parameter", expanded=True, icon=":material/tune:"):
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
    kpis([("Silhouette", fmt(m["Silhouette"]), quality(m["Silhouette"])), ("DBI (MDS)", fmt(m["DBI_MDS"]), "↓ lebih kecil lebih baik"),
          ("CHI (MDS)", fmt(m["CHI_MDS"], "{:,.0f}"), "↑ lebih besar lebih baik"), ("Balance", fmt(m["Balance"]), "1 = ukuran seimbang"),
          ("Cluster", str(int(m["Cluster"])), f"coverage {100 * m['Coverage']:.1f}%" + (f" · {int(m['Noise'])} noise" if m["Noise"] else "")),
          ('<span style="text-transform:none">η²</span> minggu nol', fmt(m["Eta2_nol"]), "tinggi = dipisah frekuensi nol")])
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
            st.subheader("Semua centroid")
            if len(r.cluster_ids):
                plot(iv.centroid_overlay_figure(r.centers, r.cluster_ids, r.center_kind, Y_TITLE[rep], config), "fv_overlay")
        with c, st.container(border=True):
            st.subheader("Jarak antar-centroid")
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
            st.subheader("Plot silhouette")
            if np.isfinite(m["Silhouette"]):
                plot(iv.fviz_silhouette_figure(r.silhouette, r.labels, codes, config), "q_sil")
                st.caption(f"{100 * np.nanmean(r.silhouette[r.labels != -1] < 0):.1f}% produk bernilai negatif (lebih dekat ke cluster lain).")
        with c, st.container(border=True):
            st.subheader("Jarak anggota ke pusat")
            plot(iv.distance_to_center_figure(r.dist_to_center, r.labels, codes, config), "q_d2c")
        if algo == "agglomerative":
            with st.container(border=True):
                st.subheader(f"Dendrogram · linkage {extra}")
                plot(iv.fviz_dend_figure(r.linkage, r.labels, int(k), config), "q_dend")
        if algo == "dbscan":
            with st.container(border=True):
                st.subheader("kNN-distance plot")
                plot(iv.knn_distance_figure(np.array(space["kdist"], dtype=float), r.eps, b["res"]["metadata"]["config"]["DBSCAN_MIN_SAMPLES"], config), "q_knn")
        if r.membership is not None:
            with st.container(border=True):
                st.subheader("Keanggotaan fuzzy")
                st.caption(f"FPC {fmt(m.get('FPC'))} · entropi partisi {fmt(m.get('Entropi_partisi'))}. Mendekati 0,5 = produk di perbatasan.")
                plot(iv.membership_stack_figure(r.membership, r.labels, codes, config), "q_mem")
        if algo != "dbscan":
            with st.container(border=True):
                st.subheader("Berapa K yang tepat?")
                rows = [{"k": kk, **b["res"]["runs"][ms.run_key(rep, algo, kk, extra)]["metrics"]} for kk in b["res"]["metadata"]["config"]["K_GRID"]]
                rows = [{**row, **{c: (np.nan if v is None else v) for c, v in row.items()}} for row in rows]
                plot(iv.fviz_nbclust_figure(rows, int(k), config), "q_nb")
    with tabs[3]:
        prof = pd.DataFrame(ix.cluster_profile_rows(r, W, codes))
        prof.insert(0, "Nama", prof["Cluster"].map(V.cluster_name))
        st.subheader("Ringkasan anggota dan produk representatif")
        st.dataframe(prof[["Nama", "Persona", "Jumlah_Produk", "Proporsi_%", "Total_Penjualan", "Porsi_Volume_%", "Produk_Representatif", "Rata2_Jarak_ke_Pusat"]],
                     hide_index=True, width="stretch", column_config={"Porsi_Volume_%": st.column_config.ProgressColumn("Porsi volume", format="%.1f%%", min_value=0, max_value=100),
                                                                       "Proporsi_%": st.column_config.NumberColumn(format="%.1f"), "Rata2_Jarak_ke_Pusat": st.column_config.NumberColumn(format="%.3f")})
        st.subheader("Profilisasi deskriptif")
        st.dataframe(prof[["Nama", "Rata2_Mingguan", "Median_Minggu_Nol", "CV_Profil", "Perubahan_13mg_%", "Tren"]], hide_index=True, width="stretch",
                     column_config={"Rata2_Mingguan": st.column_config.NumberColumn(format="%.2f"), "CV_Profil": st.column_config.NumberColumn(format="%.3f"),
                                    "Perubahan_13mg_%": st.column_config.NumberColumn("13 mg awal → akhir", format="%+.1f%%")})
        st.subheader("Interpretasi dan rekomendasi bisnis")
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
            names = [REP_LABEL[r] for r in keys]
            ari = pd.DataFrame([[adjusted_rand_score(res["runs"][keys[a]]["labels"], res["runs"][keys[c]]["labels"]) for c in keys] for a in keys], index=names, columns=names)
            st.subheader("Seberapa sama pembagian produknya? (ARI)")
            st.caption("1 = identik, 0 = tidak lebih mirip dari acak. ARI rendah = pilihan representasi benar-benar mengubah segmentasi.")
            plot(V.create_similarity_heatmap(ari, "ARI", config), "rc_ari")
        st.info("Silhouette antar-representasi tidak sepenuhnya setara (diukur di ruang jarak masing-masing). Pilih representasi sesuai tujuan: "
                "**Data asli** untuk segmentasi volume, **Min-Max / Scale** untuk segmentasi bentuk pola; perhatikan η² minggu nol.", icon=":material/lightbulb:")
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
        st.subheader("Kamus metrik")
        for term, desc in [("Silhouette", "Pemisahan vs kekompakan pada matriks FastDTW; −1…1, lebih tinggi lebih baik (> 0,25 = struktur lemah, > 0,5 = wajar)."),
                           ("DBI & CHI (MDS)", "Dihitung pada embedding MDS 5 dimensi dari matriks FastDTW. DBI ↓, CHI ↑."),
                           ("Balance", "Entropi ukuran cluster ternormalisasi; 1 = seimbang, 0 = satu cluster dominan."),
                           ("η² minggu nol", "Seberapa besar variasi jumlah minggu nol dijelaskan label; tinggi = cluster didorong frekuensi nol, bukan bentuk."),
                           ("ARI", "Kesamaan dua pembagian produk (antar-representasi / antar-seed); bukan akurasi.")]:
            st.markdown(f"**{term}**")
            st.caption(desc)
    with c, st.container(border=True):
        st.subheader("Konfigurasi pembangunan model")
        st.json({"dibangun_utc": meta["created_at_utc"], "durasi_detik": meta["runtime_s"], "jarak": meta["distance"], "konfigurasi": cfg,
                 "model_terbaik": b["res"]["best"], "stabilitas": meta.get("stability"), "versi": meta["versions"]}, expanded=False)
    files = [p for p in sorted(config.RESULTS_DIR.iterdir()) if p.is_file()]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=f"hasil_model/{p.name}")
    st.download_button("Unduh hasil_model.zip", buf.getvalue(), "hasil_model.zip", "application/zip", icon=":material/folder_zip:")
    st.warning("Hasil bersifat deskriptif/eksploratif dari data historis, bukan prediksi. Validasi interpretasi cluster dengan konteks bisnis.", icon=":material/info:")
