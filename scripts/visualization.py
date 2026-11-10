"""Seluruh definisi visualisasi Plotly untuk notebook dan dashboard Ruang Sales.

Versi 2: palet kategorikal tervalidasi (warna mengikuti nomor cluster, bukan peringkat),
tanpa grafik sumbu ganda, ditambah visual baru (Pareto, peta panas produk × minggu,
small multiples, peringkat metrik, aliran label antar-metode, grid tuning DBSCAN, dan
keanggotaan fuzzy). Semua nama fungsi lama tetap ada dengan argumen yang kompatibel
sehingga notebook tidak perlu diubah.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------------------
# Token tampilan (fallback bila `config` tidak diberikan)
# --------------------------------------------------------------------------------------
_DEFAULTS = {
    # Tema "Dark Analytics": navy + electric cyan + golden yellow (lihat config.py)
    "PLOT_PRIMARY": "#00D5F2",
    "PLOT_ACCENT": "#FFD166",
    "PLOT_PALETTE": ["#0A9CC9", "#B88A12", "#1FA884", "#8E72EA", "#E35D8A"],
    "PLOT_NOISE": "#718BA3",
    "PLOT_INK": "#EAF2FA",
    "PLOT_MUTED": "#A9BDCF",
    "PLOT_GRID": "rgba(22,73,106,0.55)",
    "PLOT_SURFACE": "#0D2540",
    "PLOT_PAPER": "#0D2540",
    "PLOT_NEUTRAL": "#4A6B86",
    "PLOT_ALERT": "#FF647C",
    "PLOT_HOVER_BG": "#0B2945",
    "PLOT_SEQUENTIAL": ["#0D2540", "#0E4A6E", "#0B7FA8", "#00B3D6", "#7FEAF8"],
    "PLOT_SEQUENTIAL_WARM": ["#0D2540", "#4A3F2A", "#8A6A26", "#C99A2E", "#FFD166"],
    "PLOT_FONT": "Inter, Plus Jakarta Sans, Arial",
}
WEEK_TICKS = [0, 8, 16, 24, 32, 40, 51]


def _cfg(config: object | None, name: str):
    """Mengambil token tampilan dari config, atau nilai bawaan versi 2."""
    return getattr(config, name, _DEFAULTS[name]) if config is not None else _DEFAULTS[name]


def cluster_color(cluster_id: int, config: object | None = None) -> str:
    """Warna tetap untuk satu nomor cluster; noise (−1) selalu abu-abu netral."""
    if int(cluster_id) == -1:
        return _cfg(config, "PLOT_NOISE")
    palette = _cfg(config, "PLOT_PALETTE")
    return palette[int(cluster_id) % len(palette)]


def cluster_name(cluster_id: int) -> str:
    """Label pendek cluster untuk legenda dan tooltip."""
    return "Noise" if int(cluster_id) == -1 else f"C{int(cluster_id)}"


def _scale(colors: list[str]) -> list[list]:
    """Mengubah daftar warna menjadi colorscale Plotly berjarak sama."""
    n = len(colors) - 1
    return [[i / n, color] for i, color in enumerate(colors)]


def _hex_to_rgba(color: str, alpha: float = 0.12) -> str:
    """Mengubah warna HEX menjadi RGBA untuk pita rentang antar-persentil."""
    value = color.lstrip("#")
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def style_figure(fig: go.Figure, title: str | None = None, height: int = 360, config: object | None = None) -> go.Figure:
    """Menerapkan gaya grafik Ruang Sales v2: latar transparan, grid tipis, legend horizontal."""
    ink, muted, grid = _cfg(config, "PLOT_INK"), _cfg(config, "PLOT_MUTED"), _cfg(config, "PLOT_GRID")
    fig.update_layout(
        template="none",
        title=dict(text=title, font=dict(size=14, color=ink)) if title else None,
        height=height,
        margin=dict(l=8, r=24, t=56 if title else 24, b=12),
        font=dict(family=_cfg(config, "PLOT_FONT"), size=12.5, color=muted),
        paper_bgcolor=_cfg(config, "PLOT_PAPER"),
        plot_bgcolor=_cfg(config, "PLOT_PAPER"),
        colorway=_cfg(config, "PLOT_PALETTE"),
        legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom", font=dict(size=12.5, color=ink), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=_cfg(config, "PLOT_HOVER_BG"), bordercolor=_cfg(config, "PLOT_PRIMARY"), font=dict(size=13, color=_cfg(config, "PLOT_INK"), family=_cfg(config, "PLOT_FONT"))),
        separators=",.",
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, automargin=True, zeroline=False, linecolor=grid, tickcolor=grid, tickfont=dict(size=12, color=muted), title_font=dict(size=12.5, color=muted))
    fig.update_yaxes(gridcolor=grid, gridwidth=1, zeroline=False, automargin=True, tickfont=dict(size=12, color=muted), title_font=dict(size=12.5, color=muted))
    return fig


def _weekly_axis(fig: go.Figure, ylabel: str, row: int | None = None, col: int | None = None) -> go.Figure:
    """Merapikan sumbu minggu agar label W0–W51 mudah dibaca."""
    kwargs = dict(row=row, col=col) if row else {}
    fig.update_xaxes(title_text="Minggu", tickmode="array", tickvals=WEEK_TICKS,
                     ticktext=[f"W{t}" for t in WEEK_TICKS], **kwargs)
    fig.update_yaxes(title_text=ylabel, **kwargs)
    return fig


def _rolling(values: Iterable[float], window: int) -> np.ndarray:
    """Rata-rata bergerak sederhana; titik awal yang belum lengkap diisi NaN."""
    return pd.Series(np.asarray(list(values), dtype=float)).rolling(window).mean().to_numpy()


# --------------------------------------------------------------------------------------
# 01 · Ringkasan
# --------------------------------------------------------------------------------------
def create_weekly_sales_figure(weekly_stats: pd.DataFrame, config: object | None = None, rolling: int = 4) -> go.Figure:
    """Tren total volume mingguan + rata-rata bergerak dan penanda puncak/terendah."""
    primary, ink = _cfg(config, "PLOT_PRIMARY"), _cfg(config, "PLOT_INK")
    x, y = weekly_stats["Minggu"], weekly_stats["Total_Penjualan"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", line=dict(width=2.4, color=primary), fill="tozeroy",
        fillcolor=_hex_to_rgba(primary, 0.10), name="Total penjualan",
        hovertemplate="W%{x}<br>Volume: %{y:,.0f}<extra></extra>",
    ))
    if rolling and rolling > 1:
        fig.add_trace(go.Scatter(
            x=x, y=_rolling(y, rolling), mode="lines", name=f"Rata-rata bergerak {rolling} minggu",
            line=dict(width=2, color=_cfg(config, "PLOT_ACCENT"), dash="dash"), hovertemplate="W%{x}<br>MA: %{y:,.0f}<extra></extra>",
        ))
    max_idx, min_idx = int(y.idxmax()), int(y.idxmin())
    fig.add_trace(go.Scatter(
        x=x.loc[[max_idx, min_idx]], y=y.loc[[max_idx, min_idx]], mode="markers+text",
        text=["Puncak", "Terendah"], textposition=["top center", "bottom right"],
        textfont=dict(color=ink, size=11), showlegend=False,
        marker=dict(size=11, color=[_cfg(config, "PLOT_ACCENT"), _cfg(config, "PLOT_PALETTE")[3]], line=dict(color=_cfg(config, "PLOT_SURFACE"), width=2)),
        hovertemplate="W%{x}<br>Volume: %{y:,.0f}<extra></extra>",
    ))
    _weekly_axis(fig, "Volume penjualan asli")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return style_figure(fig, None, 340, config)


def create_active_products_figure(weekly_stats: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Persentase produk aktif per minggu — grafik terpisah, bukan sumbu kedua."""
    color = getattr(config, "PLOT_EMERALD", "#00C9A7") if config is not None else "#00C9A7"
    fig = go.Figure(go.Scatter(
        x=weekly_stats["Minggu"], y=weekly_stats["Persen_Produk_Aktif"], mode="lines",
        line=dict(width=2, color=color), fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.10),
        name="Produk aktif", hovertemplate="W%{x}<br>Produk aktif: %{y:.1f}%<extra></extra>",
    ))
    _weekly_axis(fig, "Produk aktif (%)")
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return style_figure(fig, None, 190, config)


def create_pareto_figure(product_stats: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Kurva konsentrasi volume: % produk (urut terbesar) vs % kumulatif volume."""
    primary, accent = _cfg(config, "PLOT_PRIMARY"), _cfg(config, "PLOT_ACCENT")
    totals = np.sort(product_stats["Total volume"].to_numpy(dtype=float))[::-1]
    cumulative = 100 * np.cumsum(totals) / totals.sum()
    share = 100 * np.arange(1, len(totals) + 1) / len(totals)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=_cfg(config, "PLOT_NEUTRAL"), dash="dot", width=1),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=np.r_[0, share], y=np.r_[0, cumulative], mode="lines", line=dict(color=primary, width=2.4),
        fill="tozeroy", fillcolor=_hex_to_rgba(primary, 0.10), name="Kumulatif volume",
        hovertemplate="%{x:.1f}% produk teratas<br>= %{y:.1f}% volume<extra></extra>",
    ))
    top20 = float(np.interp(20, share, cumulative))
    p80 = float(share[np.searchsorted(cumulative, 80)])
    fig.add_trace(go.Scatter(
        x=[20, p80], y=[top20, 80], mode="markers+text", showlegend=False,
        text=[f"20% produk → {top20:.1f}% volume", f"80% volume ← {p80:.1f}% produk"],
        textposition="bottom right", textfont=dict(color=_cfg(config, "PLOT_INK"), size=11),
        marker=dict(size=10, color=accent, line=dict(color=_cfg(config, "PLOT_SURFACE"), width=2)), hoverinfo="skip",
    ))
    fig.update_xaxes(title_text="% produk (urut volume terbesar)", range=[0, 100], ticksuffix="%")
    fig.update_yaxes(title_text="% kumulatif volume", range=[0, 102], ticksuffix="%")
    return style_figure(fig, None, 380, config)


def create_top_products_figure(product_stats: pd.DataFrame, top_n: int = 10, config: object | None = None,
                               cluster_by_product: dict | None = None) -> go.Figure:
    """Bar horizontal produk dengan volume tertinggi; diwarnai cluster bila tersedia."""
    top = product_stats.nlargest(int(top_n), "Total volume").sort_values("Total volume")
    if cluster_by_product:
        colors = [cluster_color(cluster_by_product.get(code, 0), config) for code in top["Produk"]]
        clusters = [cluster_name(cluster_by_product.get(code, 0)) for code in top["Produk"]]
    else:
        colors, clusters = _cfg(config, "PLOT_PRIMARY"), ["–"] * len(top)
    fig = go.Figure(go.Bar(
        x=top["Total volume"], y=top["Produk"], orientation="h", marker=dict(color=colors, cornerradius=4),
        text=top["Total volume"], texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False,
        customdata=np.c_[clusters, top["Minggu nol"], top["Minggu puncak"]],
        hovertemplate="<b>%{y}</b><br>Total: %{x:,.0f}<br>Cluster: %{customdata[0]}<br>Minggu nol: %{customdata[1]}<br>Puncak: %{customdata[2]}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Total volume · 52 minggu", showgrid=True, gridcolor=_cfg(config, "PLOT_GRID"))
    fig.update_yaxes(title_text=None, showgrid=False)
    return style_figure(fig, None, 390, config)


def create_zero_weeks_figure(product_stats: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Distribusi banyaknya minggu tanpa penjualan pada setiap produk (0–52)."""
    counts = product_stats["Minggu nol"].value_counts().sort_index().reindex(range(53), fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, marker=dict(color=_cfg(config, "PLOT_ACCENT"), cornerradius=2),
        hovertemplate="%{x} minggu nol<br>%{y} produk<extra></extra>",
    ))
    fig.update_xaxes(title_text="Jumlah minggu tanpa penjualan", dtick=5, range=[-0.6, 52.6])
    fig.update_yaxes(title_text="Jumlah produk")
    fig.update_layout(bargap=0.12)
    return style_figure(fig, None, 390, config)


# --------------------------------------------------------------------------------------
# 02 · Eksplorasi
# --------------------------------------------------------------------------------------
def create_exploration_figures(
    product_stats: pd.DataFrame,
    W_original: np.ndarray,
    selected_product: str,
    config: object | None = None,
    labels: np.ndarray | None = None,
    compare_product: str | None = None,
    scale: str = "raw",
    X_scaled: np.ndarray | None = None,
) -> dict[str, go.Figure]:
    """Histogram log, scatter aktivitas–volume (warna cluster), dan detail deret produk."""
    values = np.asarray(W_original, dtype=float)
    codes = product_stats["Produk"].astype(str).to_numpy()
    if selected_product not in set(codes):
        selected_product = str(codes[0])
    position = int(np.flatnonzero(codes == selected_product)[0])
    y_labels = np.zeros(len(codes), dtype=int) if labels is None else np.asarray(labels, dtype=int)

    # Histogram dengan bin logaritmik (distribusi total sangat menceng).
    edges = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]
    names = ["<10", "10+", "20+", "50+", "100+", "200+", "500+", "1rb+", "2rb+", "5rb+"]
    totals = product_stats["Total volume"].to_numpy(dtype=float)
    counts = [int(((totals >= lo) & (totals < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
    hist = go.Figure(go.Bar(x=names, y=counts, marker=dict(color=_cfg(config, "PLOT_PALETTE")[2], cornerradius=4),
                            hovertemplate="Total %{x}<br>%{y} produk<extra></extra>"))
    hist.update_xaxes(title_text="Total volume per produk · 52 minggu (bin log)")
    hist.update_yaxes(title_text="Jumlah produk")
    hist = style_figure(hist, None, 360, config)

    scatter = go.Figure()
    for cluster_id in sorted(np.unique(y_labels)):
        mask = y_labels == cluster_id
        frame = product_stats.loc[mask]
        scatter.add_trace(go.Scattergl(
            x=frame["Minggu aktif (%)"], y=frame["Total volume"].clip(lower=1), mode="markers",
            name=cluster_name(cluster_id), customdata=np.c_[frame["Produk"], frame["Minggu nol"]],
            marker=dict(size=7, color=cluster_color(cluster_id, config), opacity=0.6, line=dict(color=_cfg(config, "PLOT_SURFACE"), width=0.6)),
            hovertemplate="<b>%{customdata[0]}</b><br>Aktif: %{x:.1f}%<br>Total: %{y:,.0f}<br>Minggu nol: %{customdata[1]}<extra></extra>",
        ))
    row = product_stats.iloc[position]
    scatter.add_trace(go.Scatter(
        x=[row["Minggu aktif (%)"]], y=[max(1.0, row["Total volume"])], mode="markers+text", text=[selected_product],
        textposition="top center", showlegend=False, hoverinfo="skip",
        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=_cfg(config, "PLOT_INK"), width=2.5)),
    ))
    scatter.update_xaxes(title_text="Minggu aktif (%)", ticksuffix="%")
    scatter.update_yaxes(title_text="Total volume (skala log)", type="log")
    scatter = style_figure(scatter, None, 380, config)

    source = values if scale == "raw" or X_scaled is None else np.asarray(X_scaled, dtype=float)
    weeks = np.arange(values.shape[1])
    detail = go.Figure()
    primary = _cfg(config, "PLOT_PRIMARY")
    detail.add_trace(go.Scatter(
        x=weeks, y=source[position], mode="lines+markers", name=selected_product,
        line=dict(width=2.4, color=primary), marker=dict(size=4), fill="tozeroy" if not compare_product else None,
        fillcolor=_hex_to_rgba(primary, 0.08), hovertemplate="W%{x}: %{y:,.2f}<extra>" + selected_product + "</extra>",
    ))
    if compare_product and compare_product in set(codes):
        other = int(np.flatnonzero(codes == compare_product)[0])
        detail.add_trace(go.Scatter(x=weeks, y=source[other], mode="lines", name=compare_product,
                                    line=dict(width=2.2, color=_cfg(config, "PLOT_ACCENT")),
                                    hovertemplate="W%{x}: %{y:,.2f}<extra>" + compare_product + "</extra>"))
    else:
        detail.add_trace(go.Scatter(x=weeks, y=_rolling(source[position], 4), mode="lines", name="Rata-rata bergerak 4 minggu",
                                    line=dict(width=1.5, color=_cfg(config, "PLOT_INK"), dash="dash"), hoverinfo="skip"))
    own_cluster = y_labels[position]
    median = np.median(source[y_labels == own_cluster], axis=0)
    detail.add_trace(go.Scatter(x=weeks, y=median, mode="lines", name=f"Median {cluster_name(own_cluster)}",
                                line=dict(width=1.4, color=_cfg(config, "PLOT_PALETTE")[2], dash="dot"),
                                hovertemplate="W%{x}: %{y:,.2f}<extra>median cluster</extra>"))
    _weekly_axis(detail, {"raw": "Penjualan asli", "zscore": "Nilai z-score"}.get(scale, "Nilai Min-Max"))
    detail.update_yaxes(rangemode="tozero")
    detail.update_layout(hovermode="x unified")
    detail = style_figure(detail, None, 360, config)
    return {"sales_distribution": hist, "activity_vs_volume": scatter, "product_detail": detail}


def create_sales_heatmap_figure(
    W_original: np.ndarray,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    product_codes: np.ndarray,
    mode: str = "minmax",
    config: object | None = None,
) -> go.Figure:
    """Peta panas produk × minggu, dikelompokkan per cluster lalu diurutkan dari volume terbesar."""
    W = np.asarray(W_original, dtype=float)
    X = np.asarray(X_scaled, dtype=float)
    y = np.asarray(labels, dtype=int)
    codes = np.asarray(product_codes).astype(str)
    order = np.lexsort((-W.sum(axis=1), y))
    if mode == "zscore":  # Scale: nilai z per produk; nol tetap berwarna karena z punya arti di bawah rata-rata
        z = np.clip(X[order], -2.5, 2.5)
    else:
        z = X[order] if mode == "minmax" else np.log10(1 + W[order])
        z = np.where(W[order] == 0, np.nan, z)  # nol = putih
    text = np.array([[f"{codes[i]} · {cluster_name(y[i])}"] * W.shape[1] for i in order])
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"W{i}" for i in range(W.shape[1])], y=np.arange(len(order)), customdata=W[order], text=text,
        colorscale=_scale(_cfg(config, "PLOT_SEQUENTIAL")), hoverongaps=False, showscale=True,
        colorbar=dict(title=dict(text={"minmax": "Min-Max", "zscore": "z-score"}.get(mode, "log₁₀(1+unit)"), side="right"), thickness=10, len=0.6),
        hovertemplate="%{text}<br>%{x}: %{customdata:,.0f} unit<br>nilai warna %{z:.2f}<extra></extra>",
    ))
    boundaries = np.flatnonzero(np.diff(y[order])) + 0.5
    for b in boundaries:
        fig.add_hline(y=b, line=dict(color=_cfg(config, "PLOT_INK"), width=1.5))
    start = 0
    for cluster_id in sorted(np.unique(y)):
        n = int((y == cluster_id).sum())
        fig.add_annotation(x=-0.01, xref="paper", y=start + n / 2, yref="y", xanchor="right", showarrow=False,
                           text=f"<b>{cluster_name(cluster_id)}</b><br>{n}", font=dict(size=11, color=cluster_color(cluster_id, config)))
        start += n
    fig.update_yaxes(autorange="reversed", showticklabels=False, showgrid=False, title_text=None)
    fig.update_xaxes(tickmode="array", tickvals=[f"W{t}" for t in WEEK_TICKS], showgrid=False)
    fig = style_figure(fig, None, 560, config)
    fig.update_layout(margin=dict(l=60, r=10, t=10, b=10))
    return fig


# --------------------------------------------------------------------------------------
# 03 · Segmentasi
# --------------------------------------------------------------------------------------
def create_cluster_pattern_figure(
    W_or_X: np.ndarray,
    labels: np.ndarray,
    scale_label: str,
    config: object | None = None,
    stat: str = "median",
    band: tuple[float, float] = (10, 90),
    names: dict | None = None,
) -> go.Figure:
    """Profil mingguan tiap cluster: garis median/rata-rata dengan pita persentil."""
    values = np.asarray(W_or_X, dtype=float)
    y = np.asarray(labels, dtype=int)
    weeks = np.arange(values.shape[1])
    fig = go.Figure()
    for cluster_id in sorted(np.unique(y)):
        group = values[y == cluster_id]
        center = group.mean(axis=0) if stat == "mean" else np.median(group, axis=0)
        low, high = np.percentile(group, band[0], axis=0), np.percentile(group, band[1], axis=0)
        color = cluster_color(cluster_id, config)
        label = f"{cluster_name(cluster_id)}" + (f" · {names[cluster_id]}" if names and cluster_id in names else "") + f" (n={len(group)})"
        fig.add_trace(go.Scatter(x=weeks, y=high, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", legendgroup=label))
        fig.add_trace(go.Scatter(x=weeks, y=low, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=_hex_to_rgba(color, 0.13), showlegend=False, hoverinfo="skip", legendgroup=label))
        fig.add_trace(go.Scatter(
            x=weeks, y=center, mode="lines", legendgroup=label, name=label,
            line=dict(width=2.8, color=color, dash="dash" if cluster_id == -1 else "solid"),
            hovertemplate="%{y:.3f}<extra>" + cluster_name(cluster_id) + "</extra>",
        ))
    _weekly_axis(fig, scale_label)
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return style_figure(fig, None, 400, config)


def create_cluster_small_multiples_figure(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    product_codes: np.ndarray,
    medoids: dict | None = None,
    config: object | None = None,
    sample: int = 40,
    seed: int = 42,
) -> go.Figure:
    """Satu panel per cluster: sampel anggota (garis tipis), rata-rata tebal, dan medoid putus-putus."""
    X = np.asarray(X_scaled, dtype=float)
    y = np.asarray(labels, dtype=int)
    codes = np.asarray(product_codes).astype(str)
    clusters = sorted(np.unique(y))
    rng = np.random.default_rng(seed)
    fig = make_subplots(rows=1, cols=len(clusters), shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[f"{cluster_name(c)} · {(y == c).sum()} produk" for c in clusters])
    weeks = np.arange(X.shape[1])
    for col, cluster_id in enumerate(clusters, start=1):
        idx = np.flatnonzero(y == cluster_id)
        pick = rng.choice(idx, size=min(sample, len(idx)), replace=False)
        color = cluster_color(cluster_id, config)
        for i in pick:
            fig.add_trace(go.Scattergl(x=weeks, y=X[i], mode="lines", line=dict(width=0.8, color=_hex_to_rgba(color, 0.25)),
                                       showlegend=False, hovertemplate=codes[i] + " · W%{x}: %{y:.2f}<extra></extra>"), row=1, col=col)
        fig.add_trace(go.Scatter(x=weeks, y=X[idx].mean(axis=0), mode="lines", line=dict(width=2.6, color=_cfg(config, "PLOT_INK")),
                                 name="Rata-rata", showlegend=col == 1, legendgroup="mean",
                                 hovertemplate="Rata-rata W%{x}: %{y:.3f}<extra></extra>"), row=1, col=col)
        medoid = (medoids or {}).get(cluster_id)
        if medoid in set(codes):
            m_idx = int(np.flatnonzero(codes == medoid)[0])
            fig.add_trace(go.Scatter(x=weeks, y=X[m_idx], mode="lines", line=dict(width=1.8, color=color, dash="dash"),
                                     name=f"Medoid", showlegend=col == 1, legendgroup="medoid",
                                     hovertemplate=f"Medoid {medoid} · W%{{x}}: %{{y:.2f}}<extra></extra>"), row=1, col=col)
        _weekly_axis(fig, "Min-Max" if col == 1 else "", row=1, col=col)
        fig.update_xaxes(title_text=None, row=1, col=col)
    fig.update_yaxes(range=[0, 1.02])
    fig = style_figure(fig, None, 300, config)
    fig.update_annotations(font=dict(size=12, color=_cfg(config, "PLOT_INK")))
    return fig


def create_cluster_contribution_figure(profiles: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Membandingkan proporsi jumlah produk dan proporsi volume setiap cluster."""
    labels = [cluster_name(value) for value in profiles["Cluster"]]
    palette = _cfg(config, "PLOT_PALETTE")
    fig = go.Figure()
    fig.add_bar(y=labels, x=profiles["Persen_Produk"], orientation="h", name="Porsi produk",
                marker=dict(color=palette[2], cornerradius=4), text=profiles["Persen_Produk"], texttemplate="%{text:.1f}%",
                textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1f}% produk<extra></extra>")
    fig.add_bar(y=labels, x=profiles["Porsi_Volume"], orientation="h", name="Porsi volume",
                marker=dict(color=palette[4], cornerradius=4), text=profiles["Porsi_Volume"], texttemplate="%{text:.1f}%",
                textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1f}% volume<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_xaxes(title_text="Persentase dari keseluruhan", ticksuffix="%", range=[0, 118], showgrid=True, gridcolor=_cfg(config, "PLOT_GRID"))
    fig.update_yaxes(title_text=None, autorange="reversed", showgrid=False)
    return style_figure(fig, None, 300, config)


def create_membership_figure(max_membership: np.ndarray, labels: np.ndarray, config: object | None = None) -> go.Figure:
    """Histogram derajat keanggotaan fuzzy tertinggi per produk, dipisah per cluster."""
    u = np.asarray(max_membership, dtype=float)
    y = np.asarray(labels, dtype=int)
    fig = go.Figure()
    for cluster_id in sorted(np.unique(y)):
        fig.add_trace(go.Histogram(x=u[y == cluster_id], name=cluster_name(cluster_id), xbins=dict(start=0.5, end=1.0, size=0.025),
                                   marker=dict(color=cluster_color(cluster_id, config)), opacity=0.85,
                                   hovertemplate="u %{x}<br>%{y} produk<extra>" + cluster_name(cluster_id) + "</extra>"))
    fig.add_vline(x=0.6, line=dict(color=_cfg(config, "PLOT_MUTED"), width=1, dash="dot"),
                  annotation_text="perbatasan", annotation_position="top right")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Derajat keanggotaan maksimum", range=[0.48, 1.0])
    fig.update_yaxes(title_text="Jumlah produk")
    return style_figure(fig, None, 340, config)


# --------------------------------------------------------------------------------------
# 04 · Perbandingan
# --------------------------------------------------------------------------------------
METRIC_SPECS = [
    ("Silhouette_FastDTW", "Silhouette ↑", True),
    ("DBI_Medoid_FastDTW", "DBI medoid ↓", False),
    ("CHI_MinMax_Euclidean", "CHI ↑", True),
    ("DBI_MinMax_Euclidean", "DBI Eucl. ↓", False),
    ("Coverage", "Coverage ↑", True),
    ("Eta2_Minggu_Nol", "η² nol ↓", False),
    ("Stabilitas_ARI", "Stab. ARI ↑", True),
]


def metric_ranks(metrics: pd.DataFrame) -> pd.DataFrame:
    """Peringkat per metrik (1 = terbaik) dengan arah ↑/↓ dan penanganan nilai seri."""
    ranks = pd.DataFrame(index=metrics["Metode"])
    for column, _, higher in METRIC_SPECS:
        series = metrics.set_index("Metode")[column].astype(float)
        ranks[column] = series.rank(ascending=not higher, method="min")
    return ranks


def create_metrics_comparison_figure(metrics: pd.DataFrame, config: object | None = None, highlight: str | None = None) -> go.Figure:
    """Empat indeks evaluasi utama dalam small multiples; satu metode dapat disorot."""
    highlight = highlight or getattr(config, "DEFAULT_METHOD", None)
    specs = METRIC_SPECS[:2] + [METRIC_SPECS[2], METRIC_SPECS[6]]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s[1] for s in specs], horizontal_spacing=0.18, vertical_spacing=0.2)
    for index, (metric, _, higher) in enumerate(specs):
        row, col = index // 2 + 1, index % 2 + 1
        ordered = metrics.sort_values(metric, ascending=not higher, na_position="last")
        fig.add_trace(go.Bar(
            x=ordered[metric], y=ordered["Metode"], orientation="h", showlegend=False,
            marker=dict(color=[_cfg(config, "PLOT_ACCENT") if m == highlight else _cfg(config, "PLOT_PRIMARY") for m in ordered["Metode"]], cornerradius=3),
            text=ordered[metric].map(lambda v: "tidak diuji" if pd.isna(v) else f"{v:.3f}"),
            textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
        ), row=row, col=col)
        fig.update_yaxes(autorange="reversed", showgrid=False, row=row, col=col)
        fig.update_xaxes(showgrid=True, gridcolor=_cfg(config, "PLOT_GRID"), row=row, col=col)
    fig = style_figure(fig, None, 620, config)
    fig.update_annotations(font=dict(size=12, color=_cfg(config, "PLOT_INK")))
    return fig


def create_rank_heatmap_figure(metrics: pd.DataFrame, config: object | None = None, display_names: dict | None = None) -> go.Figure:
    """Peta peringkat metode × metrik (1 gelap = terbaik), sudah memperhitungkan arah ↑/↓."""
    ranks = metric_ranks(metrics)
    names = [display_names.get(m, m) if display_names else m for m in ranks.index]
    raw = metrics.set_index("Metode")[[c for c, _, _ in METRIC_SPECS]]
    text = ranks.map(lambda v: "–" if pd.isna(v) else f"#{int(v)}")
    fig = go.Figure(go.Heatmap(
        z=ranks.to_numpy(), x=[label for _, label, _ in METRIC_SPECS], y=names, text=text.to_numpy(), texttemplate="%{text}",
        customdata=raw.to_numpy(), zmin=1, zmax=len(ranks), colorscale=_scale(_cfg(config, "PLOT_SEQUENTIAL")[::-1]),
        showscale=False, xgap=3, ygap=3, hoverongaps=False, textfont=dict(size=12),
        hovertemplate="%{y} · %{x}<br>Peringkat %{text} · nilai %{customdata:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    return style_figure(fig, None, 340, config)


def create_coverage_figure(metrics: pd.DataFrame, config: object | None = None, labels_frame: pd.DataFrame | None = None) -> go.Figure:
    """Komposisi produk per cluster untuk tiap metode (stacked 100%). Tanpa sumbu ganda.

    Jika `labels_frame` tidak diberikan (pemanggilan lama di notebook), grafik menampilkan
    coverage saja dengan jumlah cluster sebagai label teks.
    """
    methods = metrics["Metode"].tolist()
    fig = go.Figure()
    if labels_frame is None:
        fig.add_bar(y=methods, x=100 * metrics["Coverage"], orientation="h", marker=dict(color=_cfg(config, "PLOT_PRIMARY"), cornerradius=4),
                    text=[f"{100 * c:.1f}% · {k} cluster" for c, k in zip(metrics["Coverage"], metrics["Jumlah_Cluster"])],
                    textposition="inside", hovertemplate="%{y}: %{x:.1f}%<extra></extra>", name="Coverage")
    else:
        all_clusters = sorted({int(c) for m in methods for c in np.unique(labels_frame[m])}, key=lambda c: (c == -1, c))
        for cluster_id in all_clusters:
            shares = [100 * float((labels_frame[m].to_numpy(dtype=int) == cluster_id).mean()) for m in methods]
            fig.add_bar(y=methods, x=shares, orientation="h", name=cluster_name(cluster_id), marker=dict(color=cluster_color(cluster_id, config)),
                        text=[f"{s:.0f}%" if s >= 6 else "" for s in shares], textposition="inside", insidetextanchor="middle",
                        textfont=dict(color=_cfg(config, "PLOT_SURFACE")), hovertemplate="%{y} · " + cluster_name(cluster_id) + ": %{x:.1f}%<extra></extra>")
        fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Porsi produk (%)", range=[0, 100], ticksuffix="%", showgrid=True, gridcolor=_cfg(config, "PLOT_GRID"))
    fig.update_yaxes(autorange="reversed", showgrid=False, title_text=None)
    fig.update_layout(bargap=0.35)
    return style_figure(fig, None, 340, config)


def create_similarity_heatmap(matrix: pd.DataFrame, metric_name: str = "ARI", config: object | None = None) -> go.Figure:
    """Heatmap kesepakatan partisi berdasarkan ARI atau NMI (ramp satu hue)."""
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=matrix.columns.tolist(), y=matrix.index.tolist(), zmin=0, zmax=1,
        colorscale=_scale(_cfg(config, "PLOT_SEQUENTIAL")), text=matrix.round(2).to_numpy(), texttemplate="%{text:.2f}",
        textfont=dict(size=15, color=_cfg(config, "PLOT_INK")),
        xgap=3, ygap=3, colorbar=dict(title=dict(text=metric_name), thickness=10, len=0.7),
        hovertemplate="%{y} × %{x}<br>" + metric_name + " = %{z:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-25)
    return style_figure(fig, None, 420, config)


def create_label_flow_sankey(labels_a: np.ndarray, labels_b: np.ndarray, name_a: str, name_b: str, config: object | None = None) -> go.Figure:
    """Diagram aliran: ke cluster mana produk dari metode A berpindah pada metode B."""
    a, b = np.asarray(labels_a, dtype=int), np.asarray(labels_b, dtype=int)
    left, right = sorted(np.unique(a)), sorted(np.unique(b))
    nodes = [f"{name_a} · {cluster_name(c)}" for c in left] + [f"{name_b} · {cluster_name(c)}" for c in right]
    colors = [cluster_color(c, config) for c in left] + [cluster_color(c, config) for c in right]
    source, target, value, link_colors = [], [], [], []
    for i, ca in enumerate(left):
        for j, cb in enumerate(right):
            n = int(((a == ca) & (b == cb)).sum())
            if n:
                source.append(i); target.append(len(left) + j); value.append(n)
                link_colors.append(_hex_to_rgba(cluster_color(ca, config), 0.32))
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, color=colors, pad=18, thickness=14, line=dict(color=_cfg(config, "PLOT_SURFACE"), width=0.5),
                  hovertemplate="%{label}<br>%{value} produk<extra></extra>"),
        link=dict(source=source, target=target, value=value, color=link_colors,
                  hovertemplate="%{source.label} → %{target.label}<br>%{value} produk<extra></extra>"),
    ))
    return style_figure(fig, None, 340, config)


def create_tuning_k_figure(tuning_per_k: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Perubahan Silhouette dan DBI terhadap jumlah cluster aktual."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Silhouette FastDTW ↑", "DBI medoid FastDTW ↓"])
    palette = _cfg(config, "PLOT_PALETTE")
    for index, (method, subset) in enumerate(tuning_per_k.groupby("Metode")):
        subset = subset.sort_values("k_Aktual")
        color = palette[index % len(palette)]
        fig.add_trace(go.Scatter(x=subset["k_Aktual"], y=subset["Silhouette_FastDTW"], mode="lines+markers", name=method,
                                 legendgroup=method, line=dict(color=color, width=2), marker=dict(size=9)), row=1, col=1)
        fig.add_trace(go.Scatter(x=subset["k_Aktual"], y=subset["DBI_Medoid_FastDTW"], mode="lines+markers", name=method,
                                 legendgroup=method, showlegend=False, line=dict(color=color, width=2), marker=dict(size=9)), row=1, col=2)
    fig.update_xaxes(title_text="Jumlah cluster aktual (k)", dtick=1)
    fig = style_figure(fig, None, 420, config)
    fig.update_annotations(font=dict(size=12, color=_cfg(config, "PLOT_INK")))
    return fig


def create_dbscan_tuning_figure(dbscan_tuning: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Trade-off Silhouette non-noise dan proporsi noise terhadap eps DBSCAN (garis)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Silhouette non-noise vs eps", "Produk noise (%) vs eps"])
    palette = _cfg(config, "PLOT_PALETTE")
    for index, (min_samples, subset) in enumerate(dbscan_tuning.groupby("min_samples")):
        subset = subset.sort_values("eps")
        color = palette[index % len(palette)]
        fig.add_trace(go.Scatter(x=subset["eps"], y=subset["Silhouette_FastDTW"], mode="lines+markers", name=f"min_samples={int(min_samples)}",
                                 legendgroup=str(min_samples), line=dict(color=color, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=subset["eps"], y=100 * (1 - subset["Coverage"]), mode="lines+markers", name=f"min_samples={int(min_samples)}",
                                 legendgroup=str(min_samples), showlegend=False, line=dict(color=color, width=2)), row=1, col=2)
    fig.update_xaxes(title_text="eps (skala FastDTW)")
    fig = style_figure(fig, None, 420, config)
    fig.update_annotations(font=dict(size=12, color=_cfg(config, "PLOT_INK")))
    return fig


def create_dbscan_grid_figure(dbscan_tuning: pd.DataFrame, best_params: dict | None = None, config: object | None = None) -> go.Figure:
    """Grid eps-kuantil × min_samples: Silhouette (kiri) dan persentase noise (kanan); ★ = terpilih."""
    frame = dbscan_tuning.copy()
    frame["Noise_%"] = 100 * (1 - frame["Coverage"])
    sil = frame.pivot_table(index="min_samples", columns="eps_quantile", values="Silhouette_FastDTW", aggfunc="first", dropna=False)
    noise = frame.pivot_table(index="min_samples", columns="eps_quantile", values="Noise_%", aggfunc="first", dropna=False)
    noise = noise.reindex(index=sil.index, columns=sil.columns)
    rows = [f"min_samples {int(v)}" for v in sil.index]
    cols = [f"q{v:.2f}" for v in sil.columns]
    star = np.full(sil.shape, "", dtype=object)
    if best_params:
        for i, ms in enumerate(sil.index):
            for j, q in enumerate(sil.columns):
                if int(ms) == int(best_params.get("min_samples", -1)) and np.isclose(q, best_params.get("eps_quantile", -1)):
                    star[i, j] = " ★"
    sil_text = np.where(sil.isna(), "–", sil.round(2).astype(str)) + star
    noise_text = np.where(noise.isna(), "–", noise.round(0).astype("Int64").astype(str) + "%") + star
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Silhouette non-noise ↑ (– = hanya 1 cluster)", "Produk noise ↓"], horizontal_spacing=0.08)
    fig.add_trace(go.Heatmap(z=sil.to_numpy(), x=cols, y=rows, text=sil_text, texttemplate="%{text}", showscale=False, xgap=3, ygap=3,
                             colorscale=_scale(_cfg(config, "PLOT_SEQUENTIAL")), hoverongaps=False,
                             hovertemplate="%{y} · %{x}<br>Silhouette %{z:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=noise.to_numpy(), x=cols, y=rows, text=noise_text, texttemplate="%{text}", showscale=False, xgap=3, ygap=3,
                             colorscale=_scale(_cfg(config, "PLOT_SEQUENTIAL_WARM")), zmin=0, zmax=100, hoverongaps=False,
                             hovertemplate="%{y} · %{x}<br>Noise %{z:.1f}%<extra></extra>"), row=1, col=2)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showgrid=False)
    fig = style_figure(fig, None, 300, config)
    fig.update_annotations(font=dict(size=12, color=_cfg(config, "PLOT_INK")))
    return fig


def create_silhouette_distribution_figure(detail: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Histogram silhouette per produk, warna mengikuti cluster."""
    fig = go.Figure()
    for cluster_id in sorted(detail["Cluster"].unique()):
        fig.add_trace(go.Histogram(x=detail.loc[detail["Cluster"] == cluster_id, "Silhouette"], name=cluster_name(cluster_id),
                                   xbins=dict(start=-1, end=1, size=0.05), marker=dict(color=cluster_color(cluster_id, config)), opacity=0.7))
    fig.add_vline(x=0, line=dict(color=_cfg(config, "PLOT_MUTED"), width=1.2, dash="dot"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Silhouette individual · FastDTW", range=[-1, 1])
    fig.update_yaxes(title_text="Jumlah produk")
    return style_figure(fig, None, 340, config)


def create_silhouette_box_figure(detail: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Box + titik silhouette individual menurut cluster; nilai negatif = lebih dekat ke cluster lain."""
    fig = go.Figure()
    has_code = "Product_Code" in detail.columns
    for cluster_id in sorted(detail["Cluster"].unique()):
        subset = detail.loc[detail["Cluster"] == cluster_id]
        color = cluster_color(cluster_id, config)
        fig.add_trace(go.Box(
            y=subset["Silhouette"], name=cluster_name(cluster_id), boxpoints="all", jitter=0.45, pointpos=0,
            marker=dict(color=_hex_to_rgba(color, 0.5), size=4), line=dict(color=_cfg(config, "PLOT_INK"), width=1.4),
            fillcolor="rgba(0,0,0,0)", customdata=subset["Product_Code"] if has_code else None,
            hovertemplate=("%{customdata}<br>" if has_code else "") + "Silhouette %{y:.3f}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=_cfg(config, "PLOT_MUTED"), width=1.2, dash="dot"))
    fig.update_yaxes(title_text="Silhouette individual", range=[-1, 1])
    fig.update_layout(showlegend=False)
    return style_figure(fig, None, 340, config)


def create_pca_scatter_figure(pca_data: pd.DataFrame, config: object | None = None, product_codes: np.ndarray | None = None) -> go.Figure:
    """Scatter PCA dua dimensi (visual saja), warna mengikuti nomor cluster secara konsisten."""
    fig = go.Figure()
    for cluster_id in sorted(pca_data["Cluster"].unique()):
        mask = (pca_data["Cluster"] == cluster_id).to_numpy()
        subset = pca_data.loc[mask]
        codes = np.asarray(product_codes)[mask] if product_codes is not None else np.array([""] * mask.sum())
        fig.add_trace(go.Scattergl(
            x=subset["PC1"], y=subset["PC2"], mode="markers", name=cluster_name(cluster_id), customdata=codes,
            marker=dict(size=7, color=cluster_color(cluster_id, config), opacity=0.6, line=dict(color=_cfg(config, "PLOT_SURFACE"), width=0.6)),
            hovertemplate="%{customdata}<br>PC1 %{x:.2f} · PC2 %{y:.2f}<extra>" + cluster_name(cluster_id) + "</extra>",
        ))
    fig.update_xaxes(title_text="PC1 · visualisasi saja", showgrid=True, gridcolor=_cfg(config, "PLOT_GRID"))
    fig.update_yaxes(title_text="PC2 · visualisasi saja")
    return style_figure(fig, None, 400, config)


def create_weekly_diagnostics_figure(weekly_stats: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Total penjualan mingguan dan persentase produk aktif pada panel terpisah (bukan sumbu ganda)."""
    palette = _cfg(config, "PLOT_PALETTE")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=["Total penjualan mingguan", "Produk yang aktif (%)"])
    fig.add_trace(go.Scatter(x=weekly_stats["Minggu"], y=weekly_stats["Total_Penjualan"], mode="lines+markers", name="Total penjualan",
                             line=dict(color=palette[0], width=2.4), marker=dict(size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=weekly_stats["Minggu"], y=weekly_stats["Persen_Produk_Aktif"], mode="lines+markers", name="Produk aktif",
                             line=dict(color=palette[4], width=2.4), marker=dict(size=5)), row=2, col=1)
    fig.update_xaxes(title_text="Minggu", tickmode="array", tickvals=WEEK_TICKS, ticktext=[f"W{t}" for t in WEEK_TICKS], row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=1)
    fig.update_yaxes(title_text="Produk aktif (%)", range=[0, 100], row=2, col=1)
    fig.add_vrect(x0=23.5, x1=26.5, fillcolor=_hex_to_rgba(palette[3], 0.08), line_width=0,
                  annotation_text="W24–W26", annotation_position="top left")
    fig = style_figure(fig, None, 560, config)
    fig.update_annotations(font=dict(size=12, color=_cfg(config, "PLOT_INK")))
    return fig


def build_dashboard_figures(
    W_original: np.ndarray,
    product_codes: np.ndarray,
    product_stats: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    profiles: pd.DataFrame,
    metrics: pd.DataFrame,
    ari: pd.DataFrame,
    nmi: pd.DataFrame,
    tuning_per_k: pd.DataFrame,
    tuning_dbscan: pd.DataFrame,
    silhouette_detail: pd.DataFrame,
    pca_data: pd.DataFrame,
    selected_product: str,
    config: object,
) -> dict[str, go.Figure]:
    """Kumpulan grafik lengkap untuk notebook/ekspor HTML (nama kunci lama dipertahankan)."""
    lookup = dict(zip(np.asarray(product_codes).astype(str), np.asarray(labels, dtype=int)))
    figures = {
        "01_weekly_sales": create_weekly_sales_figure(weekly_stats, config),
        "02_top_products": create_top_products_figure(product_stats, 10, config, lookup),
        "03_zero_weeks": create_zero_weeks_figure(product_stats, config),
    }
    figures.update({f"04_{name}": figure for name, figure in create_exploration_figures(
        product_stats, W_original, selected_product, config, labels=labels
    ).items()})
    figures["05_cluster_original"] = create_cluster_pattern_figure(W_original, labels, "Median volume · penjualan asli", config)
    figures["06_cluster_minmax"] = create_cluster_pattern_figure(X_scaled, labels, "Median nilai Min-Max · pola ternormalisasi", config)
    figures["07_cluster_contribution"] = create_cluster_contribution_figure(profiles, config)
    figures["08_metrics_comparison"] = create_metrics_comparison_figure(metrics, config)
    figures["09_coverage"] = create_coverage_figure(metrics, config)
    figures["10_ari_heatmap"] = create_similarity_heatmap(ari, "ARI", config)
    figures["11_nmi_heatmap"] = create_similarity_heatmap(nmi, "NMI", config)
    figures["12_tuning_k"] = create_tuning_k_figure(tuning_per_k, config)
    figures["13_dbscan_tuning"] = create_dbscan_tuning_figure(tuning_dbscan, config)
    figures["14_silhouette_distribution"] = create_silhouette_distribution_figure(silhouette_detail, config)
    figures["15_silhouette_by_cluster"] = create_silhouette_box_figure(silhouette_detail, config)
    figures["16_pca_scatter"] = create_pca_scatter_figure(pca_data, config, product_codes)
    figures["17_weekly_diagnostics"] = create_weekly_diagnostics_figure(weekly_stats, config)
    # Visual baru v2
    figures["18_pareto"] = create_pareto_figure(product_stats, config)
    figures["19_active_products"] = create_active_products_figure(weekly_stats, config)
    figures["20_sales_heatmap"] = create_sales_heatmap_figure(W_original, X_scaled, labels, product_codes, "minmax", config)
    figures["21_rank_heatmap"] = create_rank_heatmap_figure(metrics, config)
    figures["22_dbscan_grid"] = create_dbscan_grid_figure(tuning_dbscan, None, config)
    return figures
