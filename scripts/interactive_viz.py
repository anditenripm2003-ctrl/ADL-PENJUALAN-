"""Visual hasil klasterisasi interaktif bergaya R (factoextra / dbscan) dengan Plotly.

Padanan R:
* ``fviz_cluster``     → :func:`fviz_cluster_figure` (titik + convex hull/elips + centroid + medoid)
* ``fviz_silhouette``  → :func:`fviz_silhouette_figure`
* ``fviz_dend``        → :func:`fviz_dend_figure`
* ``fviz_nbclust``     → :func:`fviz_nbclust_figure`
* ``kNNdistplot``      → :func:`knn_distance_figure`
* ``dtwPlotTwoWay`` / ``dtwPlotDensity`` → :func:`dtw_alignment_figure`, :func:`dtw_cost_figure`
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import ConvexHull

from scripts.visualization import WEEK_TICKS, _cfg, _hex_to_rgba, cluster_color, cluster_name, style_figure

SYMBOLS = ["circle", "triangle-up", "square", "diamond", "cross", "x", "pentagon", "star-triangle-up"]


def _symbol(cluster_id: int) -> str:
    return "circle-open" if cluster_id == -1 else SYMBOLS[int(cluster_id) % len(SYMBOLS)]


def _ellipse(points: np.ndarray, level: float = 0.95, n: int = 80) -> np.ndarray | None:
    """Elips normal multivariat (seperti ``ellipse.type = "norm"`` di factoextra)."""
    if len(points) < 3:
        return None
    cov = np.cov(points.T)
    if not np.isfinite(cov).all() or np.linalg.det(cov) <= 1e-15:
        return None
    vals, vecs = np.linalg.eigh(cov)
    radius = np.sqrt(-2 * np.log(1 - level))  # chi-square 2 df
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.c_[np.cos(t), np.sin(t)] * np.sqrt(np.maximum(vals, 0)) * radius
    return circle @ vecs.T + points.mean(axis=0)


def fviz_cluster_figure(
    coords: np.ndarray,
    explained: np.ndarray,
    labels: np.ndarray,
    codes: np.ndarray,
    center_xy: np.ndarray,
    cluster_ids: np.ndarray,
    medoids: np.ndarray,
    center_kind: str,
    shape: str = "hull",
    spider: bool = False,
    config: object | None = None,
    axis_name: str = "Dim",
) -> go.Figure:
    """Peta cluster 2D: titik per cluster (warna + simbol), hull/elips, centroid, dan medoid."""
    xy = np.asarray(coords)[:, :2]
    y = np.asarray(labels, dtype=int)
    codes = np.asarray(codes).astype(str)
    fig = go.Figure()
    for cluster_id in sorted(np.unique(y)):
        mask = y == cluster_id
        pts = xy[mask]
        color = cluster_color(cluster_id, config)
        if cluster_id != -1 and shape != "none" and len(pts) >= 3:
            outline = None
            if shape == "hull":
                try:
                    hull = ConvexHull(pts)
                    outline = pts[np.r_[hull.vertices, hull.vertices[0]]]
                except Exception:
                    outline = None
            else:
                outline = _ellipse(pts)
            if outline is not None:
                fig.add_trace(go.Scatter(x=outline[:, 0], y=outline[:, 1], mode="lines", fill="toself",
                                         fillcolor=_hex_to_rgba(color, 0.10), line=dict(color=color, width=1.2),
                                         hoverinfo="skip", showlegend=False, legendgroup=str(cluster_id)))
        if spider and cluster_id != -1:
            pos = int(np.flatnonzero(np.asarray(cluster_ids) == cluster_id)[0])
            cx, cy = center_xy[pos]
            sx = np.c_[pts[:, 0], np.full(len(pts), cx), np.full(len(pts), np.nan)].ravel()
            sy = np.c_[pts[:, 1], np.full(len(pts), cy), np.full(len(pts), np.nan)].ravel()
            fig.add_trace(go.Scattergl(x=sx, y=sy, mode="lines", line=dict(color=_hex_to_rgba(color, 0.18), width=0.7),
                                       hoverinfo="skip", showlegend=False, legendgroup=str(cluster_id)))
        fig.add_trace(go.Scattergl(
            x=pts[:, 0], y=pts[:, 1], mode="markers", name=f"{cluster_name(cluster_id)} · {mask.sum()}",
            legendgroup=str(cluster_id), customdata=codes[mask],
            marker=dict(size=7, color=color if cluster_id != -1 else _cfg(config, "PLOT_NOISE"), symbol=_symbol(cluster_id),
                        opacity=0.7, line=dict(color="#fff", width=0.5)),
            hovertemplate="<b>%{customdata}</b><br>" + cluster_name(cluster_id) + "<extra></extra>",
        ))
    if len(cluster_ids):
        kind_label = {"mean": "Centroid (rata-rata)", "dba": "Centroid DBA", "medoid": "Medoid"}[center_kind]
        fig.add_trace(go.Scatter(
            x=center_xy[:, 0], y=center_xy[:, 1], mode="markers+text", name=kind_label,
            text=[f"<b>{cluster_name(c)}</b>" for c in cluster_ids], textposition="top center",
            textfont=dict(size=13, color=_cfg(config, "PLOT_INK")),
            marker=dict(size=20, symbol="diamond" if center_kind != "medoid" else "star", color=[cluster_color(c, config) for c in cluster_ids],
                        line=dict(color=_cfg(config, "PLOT_INK"), width=2)),
            customdata=[codes[m] for m in medoids],
            hovertemplate="<b>%{text}</b><br>" + kind_label + "<br>Medoid: %{customdata}<extra></extra>",
        ))
        if center_kind != "medoid":
            fig.add_trace(go.Scatter(
                x=xy[medoids, 0], y=xy[medoids, 1], mode="markers", name="Medoid (produk representatif)",
                marker=dict(size=13, symbol="star", color="#fff", line=dict(color=_cfg(config, "PLOT_INK"), width=1.6)),
                customdata=codes[medoids], hovertemplate="Medoid %{customdata}<extra></extra>",
            ))
    ex = list(explained) + [0, 0]
    fig.update_xaxes(title_text=f"{axis_name}1 ({100 * ex[0]:.1f}%)", showgrid=True, gridcolor=_cfg(config, "PLOT_GRID"),
                     zeroline=True, zerolinecolor="#D7D2C4")
    fig.update_yaxes(title_text=f"{axis_name}2 ({100 * ex[1]:.1f}%)", zeroline=True, zerolinecolor="#D7D2C4")
    return style_figure(fig, None, 520, config)


def centroid_series_figure(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster_ids: np.ndarray,
    medoids: np.ndarray,
    codes: np.ndarray,
    center_kind: str,
    y_title: str,
    config: object | None = None,
    sample: int = 50,
    seed: int = 7,
) -> go.Figure:
    """Satu panel per cluster: anggota (garis tipis), pita IQR, centroid tebal, medoid putus-putus."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(labels, dtype=int)
    codes = np.asarray(codes).astype(str)
    ids = list(cluster_ids)
    cols = min(len(ids), 3) or 1
    rows = int(np.ceil(len(ids) / cols)) or 1
    fig = make_subplots(rows=rows, cols=cols, shared_yaxes=True, horizontal_spacing=0.04, vertical_spacing=0.16,
                        subplot_titles=[f"{cluster_name(c)} · {(y == c).sum()} produk" for c in ids])
    rng = np.random.default_rng(seed)
    weeks = np.arange(X.shape[1])
    center_name = {"mean": "Centroid (rata-rata)", "dba": "Centroid DBA", "medoid": "Medoid = centroid"}[center_kind]
    for index, c in enumerate(ids):
        r, col = index // cols + 1, index % cols + 1
        members = np.flatnonzero(y == c)
        color = cluster_color(c, config)
        pick = rng.choice(members, size=min(sample, len(members)), replace=False)
        xs = np.tile(np.r_[weeks, np.nan], len(pick))
        ys = np.c_[X[pick], np.full(len(pick), np.nan)].ravel()
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color=_hex_to_rgba(color, 0.22), width=0.8),
                                   hoverinfo="skip", showlegend=False), row=r, col=col)
        q1, q3 = np.percentile(X[members], 25, axis=0), np.percentile(X[members], 75, axis=0)
        fig.add_trace(go.Scatter(x=weeks, y=q3, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False), row=r, col=col)
        fig.add_trace(go.Scatter(x=weeks, y=q1, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=_hex_to_rgba(color, 0.18),
                                 hoverinfo="skip", showlegend=index == 0, name="IQR anggota", legendgroup="iqr"), row=r, col=col)
        fig.add_trace(go.Scatter(x=weeks, y=centers[index], mode="lines", line=dict(color=_cfg(config, "PLOT_INK"), width=3),
                                 name=center_name, legendgroup="center", showlegend=index == 0,
                                 hovertemplate=f"{cluster_name(c)} W%{{x}}: %{{y:.3f}}<extra>{center_name}</extra>"), row=r, col=col)
        if center_kind != "medoid":
            fig.add_trace(go.Scatter(x=weeks, y=X[medoids[index]], mode="lines", line=dict(color=color, width=2, dash="dash"),
                                     name="Medoid", legendgroup="medoid", showlegend=index == 0,
                                     hovertemplate=f"Medoid {codes[medoids[index]]} W%{{x}}: %{{y:.3f}}<extra></extra>"), row=r, col=col)
        fig.update_xaxes(tickmode="array", tickvals=WEEK_TICKS, ticktext=[f"W{t}" for t in WEEK_TICKS], row=r, col=col)
    fig.update_yaxes(title_text=y_title, col=1)
    fig = style_figure(fig, None, 300 * rows + 40, config)
    fig.update_annotations(font=dict(size=12, color=_cfg(config, "PLOT_INK")))
    return fig


def centroid_overlay_figure(centers: np.ndarray, cluster_ids: np.ndarray, center_kind: str, y_title: str,
                            config: object | None = None) -> go.Figure:
    """Semua centroid dalam satu grafik untuk membandingkan bentuk pola antar-cluster."""
    fig = go.Figure()
    weeks = np.arange(np.asarray(centers).shape[1])
    for pos, c in enumerate(cluster_ids):
        fig.add_trace(go.Scatter(x=weeks, y=centers[pos], mode="lines+markers", name=cluster_name(c),
                                 line=dict(color=cluster_color(c, config), width=2.6), marker=dict(size=4),
                                 hovertemplate="W%{x}: %{y:.3f}<extra>" + cluster_name(c) + "</extra>"))
    fig.update_xaxes(title_text="Minggu", tickmode="array", tickvals=WEEK_TICKS, ticktext=[f"W{t}" for t in WEEK_TICKS])
    fig.update_yaxes(title_text=y_title)
    fig.update_layout(hovermode="x unified")
    return style_figure(fig, None, 340, config)


def fviz_silhouette_figure(silhouette: np.ndarray, labels: np.ndarray, codes: np.ndarray, config: object | None = None) -> go.Figure:
    """Plot silhouette gaya R: batang terurut per cluster dan garis rata-rata putus-putus."""
    s = np.asarray(silhouette, dtype=float)
    y = np.asarray(labels, dtype=int)
    codes = np.asarray(codes).astype(str)
    keep = np.isfinite(s) & (y != -1)
    frame = pd.DataFrame({"s": s[keep], "c": y[keep], "code": codes[keep]}).sort_values(["c", "s"], ascending=[True, False])
    frame["x"] = np.arange(len(frame))
    fig = go.Figure()
    for c, part in frame.groupby("c"):
        fig.add_trace(go.Bar(x=part["x"], y=part["s"], name=f"{cluster_name(c)} · rata² {part['s'].mean():.2f}",
                             marker=dict(color=cluster_color(c, config), line=dict(width=0)), customdata=part["code"],
                             hovertemplate="%{customdata}<br>Silhouette %{y:.3f}<extra></extra>"))
    if len(frame):
        fig.add_hline(y=frame["s"].mean(), line=dict(color="#C0392B", dash="dash", width=1.5),
                      annotation_text=f"rata-rata {frame['s'].mean():.3f}", annotation_position="top right")
    fig.update_layout(bargap=0)
    fig.update_xaxes(showticklabels=False, title_text="Produk (diurutkan per cluster)")
    fig.update_yaxes(title_text="Lebar silhouette Si", range=[min(-0.2, float(frame["s"].min()) - 0.05) if len(frame) else -1, 1])
    return style_figure(fig, None, 360, config)


def fviz_dend_figure(linkage_matrix: np.ndarray, labels: np.ndarray, k: int, config: object | None = None) -> go.Figure:
    """Dendrogram berwarna per cluster + garis potong (seperti ``fviz_dend(k = ...)``)."""
    Z = np.asarray(linkage_matrix)
    n = len(Z) + 1
    y = np.asarray(labels, dtype=int)
    leaves = dendrogram(Z, no_plot=True, count_sort=True)["leaves"]
    xpos = {leaf: 5.0 + 10.0 * i for i, leaf in enumerate(leaves)}
    height = {i: 0.0 for i in range(n)}
    node_cluster = {i: int(y[i]) for i in range(n)}
    segments: dict[int, tuple[list, list]] = {}
    for idx, (a, b, h, _) in enumerate(Z):
        a, b, node = int(a), int(b), n + idx
        xpos[node] = (xpos[a] + xpos[b]) / 2
        height[node] = float(h)
        ca, cb = node_cluster[a], node_cluster[b]
        node_cluster[node] = ca if ca == cb else -2
        bucket = segments.setdefault(node_cluster[node], ([], []))
        bucket[0].extend([xpos[a], xpos[a], xpos[b], xpos[b], None])
        bucket[1].extend([height[a], h, h, height[b], None])
    fig = go.Figure()
    for c, (xs, ds) in sorted(segments.items()):
        color = "#B9B3A4" if c == -2 else cluster_color(c, config)
        fig.add_trace(go.Scatter(x=xs, y=ds, mode="lines", line=dict(color=color, width=1.1), hoverinfo="skip",
                                 name="di atas potongan" if c == -2 else cluster_name(c), showlegend=c != -2))
    if k >= 2 and len(Z) >= k:
        cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2 if k > 1 else Z[-1, 2]
        fig.add_hline(y=cut, line=dict(color="#C0392B", dash="dash", width=1.4), annotation_text=f"potong k = {k}",
                      annotation_position="top left")
    fig.update_xaxes(showticklabels=False, title_text=f"{n} produk (urutan daun dendrogram)", range=[0, 10 * n])
    fig.update_yaxes(title_text="Tinggi penggabungan (jarak)")
    return style_figure(fig, None, 420, config)


def fviz_nbclust_figure(rows: list[dict], chosen_k: int | None = None, config: object | None = None) -> go.Figure:
    """Metode siku (total jarak ke pusat) + rata-rata silhouette terhadap k."""
    frame = pd.DataFrame(rows)
    best_k = int(frame.loc[frame["Silhouette"].idxmax(), "k"]) if frame["Silhouette"].notna().any() else None
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Metode siku · total jarak ke pusat ↓", "Rata-rata silhouette ↑"])
    primary = _cfg(config, "PLOT_PRIMARY")
    fig.add_trace(go.Scatter(x=frame["k"], y=frame["Within"], mode="lines+markers", line=dict(color=primary, width=2.4),
                             marker=dict(size=9), showlegend=False, hovertemplate="k=%{x}<br>%{y:,.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame["k"], y=frame["Silhouette"], mode="lines+markers", line=dict(color=primary, width=2.4),
                             marker=dict(size=9), showlegend=False, hovertemplate="k=%{x}<br>%{y:.3f}<extra></extra>"), row=1, col=2)
    for k, color, label in [(best_k, "#C0392B", "terbaik"), (chosen_k, _cfg(config, "PLOT_ACCENT"), "dipilih")]:
        if k is not None:
            for col in (1, 2):
                fig.add_vline(x=k, line=dict(color=color, dash="dash", width=1.3), row=1, col=col)
    fig.update_xaxes(title_text="Jumlah cluster k", dtick=1)
    fig = style_figure(fig, None, 340, config)
    fig.update_annotations(font=dict(size=12, color=_cfg(config, "PLOT_INK")))
    return fig


def knn_distance_figure(kdist: np.ndarray, eps: float | None, k: int, config: object | None = None) -> go.Figure:
    """Plot jarak tetangga ke-k terurut (``kNNdistplot``) dengan garis eps terpilih."""
    fig = go.Figure(go.Scatter(x=np.arange(len(kdist)), y=kdist, mode="lines", line=dict(color=_cfg(config, "PLOT_PRIMARY"), width=2.2),
                               hovertemplate="titik ke-%{x}<br>jarak %{y:.4g}<extra></extra>", name=f"jarak tetangga ke-{k}"))
    if eps is not None:
        fig.add_hline(y=eps, line=dict(color="#C0392B", dash="dash", width=1.5), annotation_text=f"eps = {eps:.4g}",
                      annotation_position="top left")
    fig.update_xaxes(title_text="Produk diurutkan menurut jarak")
    fig.update_yaxes(title_text=f"Jarak ke tetangga ke-{k}")
    return style_figure(fig, None, 320, config)


def distance_to_center_figure(dist_to_center: np.ndarray, labels: np.ndarray, codes: np.ndarray, config: object | None = None) -> go.Figure:
    """Sebaran jarak setiap produk ke pusat cluster-nya; ekor panjang = anggota pinggiran."""
    fig = go.Figure()
    y = np.asarray(labels, dtype=int)
    codes = np.asarray(codes).astype(str)
    for c in sorted(set(y.tolist()) - {-1}):
        mask = y == c
        color = cluster_color(c, config)
        fig.add_trace(go.Box(y=dist_to_center[mask], name=cluster_name(c), boxpoints="outliers", customdata=codes[mask],
                             marker=dict(color=color, size=4), line=dict(color=color), fillcolor=_hex_to_rgba(color, 0.15),
                             hovertemplate="%{customdata}<br>jarak %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="Jarak ke pusat cluster")
    fig.update_layout(showlegend=False)
    return style_figure(fig, None, 330, config)


def center_distance_figure(center_matrix: np.ndarray, cluster_ids: np.ndarray, config: object | None = None) -> go.Figure:
    """Jarak antar-pusat cluster (makin jauh = makin terpisah)."""
    names = [cluster_name(c) for c in cluster_ids]
    fig = go.Figure(go.Heatmap(z=center_matrix, x=names, y=names, text=np.round(center_matrix, 3), texttemplate="%{text}",
                               colorscale=[[0, "#F5F3EC"], [1, _cfg(config, "PLOT_PRIMARY")]], showscale=False, xgap=3, ygap=3,
                               hovertemplate="%{y} ↔ %{x}: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return style_figure(fig, None, 260, config)


def cluster_heatmap_figure(X: np.ndarray, labels: np.ndarray, dist_to_center: np.ndarray, codes: np.ndarray,
                           config: object | None = None) -> go.Figure:
    """Peta panas anggota per cluster, diurutkan dari yang paling dekat ke pusat."""
    y = np.asarray(labels, dtype=int)
    d = np.where(np.isfinite(dist_to_center), dist_to_center, np.inf)
    order = np.lexsort((d, np.where(y == -1, 10 ** 6, y)))
    codes = np.asarray(codes).astype(str)
    fig = go.Figure(go.Heatmap(z=np.asarray(X)[order], x=[f"W{i}" for i in range(np.asarray(X).shape[1])], y=np.arange(len(order)),
                               text=np.repeat([[f"{codes[i]} · {cluster_name(y[i])}"] for i in order], np.asarray(X).shape[1], axis=1),
                               colorscale=[[i / 4, c] for i, c in enumerate(_cfg(config, "PLOT_SEQUENTIAL"))],
                               colorbar=dict(thickness=10, len=0.6), hovertemplate="%{text}<br>%{x}: %{z:.3f}<extra></extra>"))
    for b in np.flatnonzero(np.diff(y[order])) + 0.5:
        fig.add_hline(y=b, line=dict(color=_cfg(config, "PLOT_INK"), width=1.5))
    fig.update_yaxes(autorange="reversed", showticklabels=False, showgrid=False)
    fig.update_xaxes(tickmode="array", tickvals=[f"W{t}" for t in WEEK_TICKS], showgrid=False)
    return style_figure(fig, None, 480, config)


def dtw_alignment_figure(a: np.ndarray, b: np.ndarray, path: np.ndarray, name_a: str, name_b: str,
                         y_title: str, config: object | None = None) -> go.Figure:
    """Dua deret dengan garis penyejajaran DTW (``dtwPlotTwoWay``); deret B digeser agar terbaca."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    offset = (np.nanmax(a) - np.nanmin(b)) * 1.15 + 1e-9
    fig = go.Figure()
    xs, ys = [], []
    for i, j in path:
        xs += [i, j, None]
        ys += [a[i], b[j] - offset, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#B9B3A4", width=0.8), hoverinfo="skip", name="Pasangan minggu"))
    fig.add_trace(go.Scatter(x=np.arange(len(a)), y=a, mode="lines+markers", name=name_a, line=dict(color=cluster_color(0, config), width=2.2),
                             marker=dict(size=5), hovertemplate="W%{x}: %{y:.3f}<extra>" + name_a + "</extra>"))
    fig.add_trace(go.Scatter(x=np.arange(len(b)), y=b - offset, mode="lines+markers", name=f"{name_b} (digeser)",
                             line=dict(color=cluster_color(1, config), width=2.2), marker=dict(size=5), customdata=b,
                             hovertemplate="W%{x}: %{customdata:.3f}<extra>" + name_b + "</extra>"))
    fig.update_xaxes(title_text="Minggu", tickmode="array", tickvals=WEEK_TICKS, ticktext=[f"W{t}" for t in WEEK_TICKS])
    fig.update_yaxes(title_text=y_title, showticklabels=False)
    return style_figure(fig, None, 420, config)


def dtw_cost_figure(cost: np.ndarray, path: np.ndarray, window: int | None, name_a: str, name_b: str,
                    config: object | None = None) -> go.Figure:
    """Matriks biaya kumulatif DTW + jalur optimal + diagonal (Euclidean) dan pita Sakoe-Chiba."""
    C = np.where(np.isfinite(cost), cost, np.nan)
    n, m = C.shape
    fig = go.Figure(go.Heatmap(z=np.sqrt(C), colorscale=[[i / 4, c] for i, c in enumerate(_cfg(config, "PLOT_SEQUENTIAL"))],
                               colorbar=dict(title=dict(text="√biaya"), thickness=10, len=0.6),
                               hovertemplate=f"{name_a} W%{{y}} ↔ {name_b} W%{{x}}<br>√biaya kumulatif %{{z:.3f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, m - 1], y=[0, n - 1], mode="lines", line=dict(color="#6B7770", dash="dot", width=1),
                             name="Diagonal (Euclidean)", hoverinfo="skip"))
    if window is not None:
        idx = np.arange(n)
        fig.add_trace(go.Scatter(x=np.r_[idx + window, (idx - window)[::-1]].clip(0, m - 1), y=np.r_[idx, idx[::-1]], mode="lines",
                                 line=dict(color=_cfg(config, "PLOT_ACCENT"), width=1, dash="dash"), name=f"Pita ±{window}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=path[:, 1], y=path[:, 0], mode="lines", line=dict(color="#C0392B", width=2.6), name="Jalur DTW optimal",
                             hovertemplate=f"{name_a} W%{{y}} ↔ {name_b} W%{{x}}<extra></extra>"))
    fig.update_xaxes(title_text=f"Minggu {name_b}", range=[-0.5, m - 0.5], showgrid=False)
    fig.update_yaxes(title_text=f"Minggu {name_a}", range=[-0.5, n - 0.5], showgrid=False, scaleanchor="x")
    return style_figure(fig, None, 460, config)


def membership_stack_figure(membership: np.ndarray, labels: np.ndarray, codes: np.ndarray, config: object | None = None) -> go.Figure:
    """Keanggotaan fuzzy semua produk sebagai batang bertumpuk (diurutkan per cluster & keyakinan)."""
    U = np.asarray(membership, dtype=float)
    y = np.asarray(labels, dtype=int)
    codes = np.asarray(codes).astype(str)
    order = np.lexsort((-U.max(axis=1), y))
    fig = go.Figure()
    for c in range(U.shape[1]):
        fig.add_trace(go.Bar(x=np.arange(len(order)), y=U[order, c], name=f"u {cluster_name(c)}", marker=dict(color=cluster_color(c, config), line=dict(width=0)),
                             customdata=codes[order], hovertemplate="%{customdata}<br>u = %{y:.3f}<extra>" + cluster_name(c) + "</extra>"))
    fig.update_layout(barmode="stack", bargap=0)
    fig.add_hline(y=0.5, line=dict(color=_cfg(config, "PLOT_INK"), dash="dot", width=1))
    fig.update_xaxes(showticklabels=False, title_text="Produk (urut cluster, lalu keyakinan)")
    fig.update_yaxes(title_text="Derajat keanggotaan", range=[0, 1])
    return style_figure(fig, None, 320, config)
