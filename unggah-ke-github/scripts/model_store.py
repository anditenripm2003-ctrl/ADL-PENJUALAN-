"""Membangun, menyimpan, dan memuat hasil model klasterisasi (FastDTW · 3 representasi).

Dipakai dua sisi:
* **Notebook Colab** → :func:`build_model_results` lalu :func:`export_results` → ``hasil_model.zip``.
* **Dashboard Streamlit** → :func:`load_results` + :func:`run_to_result` (tanpa menghitung ulang).

Struktur ``model_results.json``::

    {"metadata": {...}, "codes": [...], "W": [[...]],
     "spaces": {rep: {"xy", "explained", "kdist", "linkage": {lk: {"Z", "leaves"}}, "note"}},
     "runs":   {"rep|algo|k|extra": {"labels", "silhouette", "dist_to_center", "centers", "center_kind",
                                     "medoids", "cluster_ids", "center_xy", "center_dm", "membership",
                                     "eps", "notes", "metrics", "n_iter"}},
     "best":   {rep: run_key}}
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from scripts import interactive as ix

ALGORITHMS = ["kmeans", "kmedoids", "agglomerative", "dbscan", "fcm", "fcmdd"]


def run_key(rep: str, algorithm: str, k: int | None = None, extra: str | float | None = None) -> str:
    """Kunci unik satu konfigurasi, mis. ``minmax|agglomerative|3|average`` atau ``zscore|dbscan|0|85``."""
    return f"{rep}|{algorithm}|{0 if algorithm == 'dbscan' else int(k)}|{'' if extra is None else extra}"


def _f(values, digits: int = 4):
    """Array → list angka JSON (NaN → None)."""
    arr = np.asarray(values, dtype=float)
    out = np.round(arr, digits).astype(object)
    out[~np.isfinite(arr)] = None
    return out.tolist()


def _clean(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _center_distances(result: ix.ClusterResult, D: np.ndarray) -> np.ndarray:
    """Jarak antar-pusat cluster: medoid → dari matriks FastDTW; DBA → DTW antar-barycenter."""
    k = len(result.cluster_ids)
    if k == 0:
        return np.zeros((0, 0))
    if result.center_kind == "medoid":
        return D[np.ix_(result.medoids, result.medoids)]
    return ix.dtw_batch(np.repeat(result.centers, k, axis=0), np.tile(result.centers, (k, 1)), None).reshape(k, k)


def _pack(result: ix.ClusterResult, D: np.ndarray) -> dict:
    return {
        "labels": result.labels.astype(int).tolist(),
        "silhouette": _f(result.silhouette, 4),
        "dist_to_center": _f(result.dist_to_center, 4),
        "centers": _f(result.centers, 4),
        "center_kind": result.center_kind,
        "medoids": [int(m) for m in result.medoids],
        "cluster_ids": [int(c) for c in result.cluster_ids],
        "center_xy": _f(result.center_xy, 4),
        "center_dm": _f(_center_distances(result, D), 4),
        "membership": _f(result.membership, 4) if result.membership is not None else None,
        "eps": _clean(result.eps),
        "notes": list(result.notes),
        "metrics": {k: _clean(v) for k, v in result.metrics.items()},
        "n_iter": int(result.n_iter),
    }


def select_best(runs: dict, rep: str, n_products: int, cfg) -> tuple[str | None, pd.DataFrame]:
    """Model terbaik satu representasi: lolos guardrail lalu peringkat rata-rata Silhouette↑, DBI↓, CHI↑."""
    rows = []
    for key, run in runs.items():
        if not key.startswith(rep + "|"):
            continue
        m = run["metrics"]
        ok = (m["Cluster"] or 0) >= cfg.MIN_CLUSTERS and (m["Balance"] or 0) >= cfg.MIN_BALANCE \
            and (m["Noise"] or 0) / n_products <= cfg.MAX_NOISE_SHARE and m["Silhouette"] is not None
        rows.append({"Kunci": key, "Lolos_guardrail": ok, **{c: m.get(c) for c in ["Silhouette", "DBI_MDS", "CHI_MDS", "Balance", "Cluster", "Noise", "Eta2_nol", "Waktu_s"]}})
    table = pd.DataFrame(rows)
    passed = table[table["Lolos_guardrail"]].copy()
    if passed.empty:
        return None, table
    ranked = pd.DataFrame(ix.rank_methods(passed.drop(columns="Lolos_guardrail").to_dict("records")))
    return str(ranked.iloc[0]["Kunci"]), ranked


def seed_stability(X: np.ndarray, D: np.ndarray, key: str, cfg, seeds=(42, 52, 62)) -> dict:
    """ARI rata-rata antar-seed untuk model terbaik (hanya metode dengan inisialisasi acak)."""
    from itertools import combinations
    from sklearn.metrics import adjusted_rand_score
    rep, algo, k, extra = key.split("|")
    if algo in ("kmedoids", "agglomerative", "dbscan"):
        return {"ARI": None, "status": "deterministik (tidak bergantung seed)"}
    labels = [ix.run_clustering(X, D, algo, cfg.DISTANCE, k=int(k), fuzziness=cfg.FUZZINESS, seed=s, n_init=1).labels for s in seeds]
    ari = float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]))
    return {"ARI": round(ari, 4), "status": f"ARI {ari:.3f} " + ("(stabil)" if ari >= 0.75 else "(kurang stabil, < 0,75)")}


def build_model_results(W: np.ndarray, codes: np.ndarray, cfg, cache_dir: str | Path | None = None,
                        log: Callable[[str], None] = print) -> dict:
    """Menjalankan seluruh grid: 3 representasi × FastDTW × 6 algoritma × K/linkage/eps."""
    W = np.asarray(W, dtype=float)
    codes = np.asarray(codes).astype(str)
    started = time.time()
    spaces, runs, best, rankings, stability = {}, {}, {}, {}, {}
    for rep in cfg.REPRESENTATIONS:
        t0 = time.time()
        X = ix.represent(W, rep)
        D, note = ix.distance_matrix(X, cfg.DISTANCE, radius=cfg.FASTDTW_RADIUS, cache_dir=cache_dir)
        emb = ix.classical_mds(D, 5)
        log(f"[{rep}] matriks {ix.DISTANCES[cfg.DISTANCE]} 811×811 siap ({time.time() - t0:.1f} dtk · {note})")
        linkages = {}
        for lk in cfg.LINKAGES:
            res = ix.run_clustering(X, D, "agglomerative", cfg.DISTANCE, k=2, linkage_method=lk)
            linkages[lk] = {"Z": _f(res.linkage, 6), "leaves": dendrogram(res.linkage, no_plot=True, count_sort=True)["leaves"]}
        spaces[rep] = {"xy": _f(emb.coords[:, :2], 4), "explained": _f(emb.explained, 4),
                       "kdist": _f(ix.kdistance(D, cfg.DBSCAN_MIN_SAMPLES), 4), "linkage": linkages, "note": note}
        specs = [(a, k, None, {}) for a in ["kmeans", "kmedoids", "fcm", "fcmdd"] for k in cfg.K_GRID]
        specs += [("agglomerative", k, lk, {"linkage_method": lk}) for lk in cfg.LINKAGES for k in cfg.K_GRID]
        specs += [("dbscan", 0, p, {"min_samples": cfg.DBSCAN_MIN_SAMPLES, "eps_percentile": p}) for p in cfg.DBSCAN_EPS_PERCENTILES]
        for i, (algo, k, extra, kwargs) in enumerate(specs, start=1):
            t1 = time.time()
            res = ix.run_clustering(X, D, algo, cfg.DISTANCE, k=max(k, 2), fuzziness=cfg.FUZZINESS, seed=cfg.RANDOM_STATE,
                                    n_init=cfg.KMEANS_N_INIT, **kwargs)
            res = ix.evaluate(res, D, emb, W)
            key = run_key(rep, algo, k, extra)
            runs[key] = _pack(res, D)
            sil = res.metrics["Silhouette"]
            log(f"  {i:2d}/{len(specs)} {key:32s} silhouette={'–' if not np.isfinite(sil) else f'{sil:.3f}'} "
                f"balance={res.metrics['Balance']:.2f} · {time.time() - t1:.1f} dtk")
        best[rep], rankings[rep] = select_best(runs, rep, len(W), cfg)
        log(f"[{rep}] model terbaik: {best[rep]}")
        if best[rep]:
            stability[rep] = seed_stability(X, D, best[rep], cfg)
            log(f"[{rep}] stabilitas antar-seed: {stability[rep]['status']}")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_products": int(W.shape[0]), "n_weeks": int(W.shape[1]), "total_sales": float(W.sum()),
        "data_digest": hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()[:16],
        "distance": {"name": ix.DISTANCES[cfg.DISTANCE], "radius": cfg.FASTDTW_RADIUS,
                     "notes": {rep: spaces[rep]["note"] for rep in spaces}},
        "config": {k: getattr(cfg, k) for k in ["REPRESENTATIONS", "K_GRID", "LINKAGES", "DBSCAN_MIN_SAMPLES",
                                                 "DBSCAN_EPS_PERCENTILES", "FUZZINESS", "RANDOM_STATE",
                                                 "MIN_BALANCE", "MAX_NOISE_SHARE", "MIN_CLUSTERS"]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "runtime_s": round(time.time() - started, 1),
        "stability": stability,
    }
    return {"metadata": metadata, "codes": codes.tolist(), "W": W.astype(int if np.allclose(W, np.round(W)) else float).tolist(),
            "spaces": spaces, "runs": runs, "best": best,
            "_rankings": {rep: table.to_dict("records") for rep, table in rankings.items()}}


def metrics_table(results: dict) -> pd.DataFrame:
    """Tabel metrik semua konfigurasi (satu baris per run)."""
    rows = []
    for key, run in results["runs"].items():
        rep, algo, k, extra = key.split("|")
        m = run["metrics"]
        rows.append({"Representasi": ix.REPRESENTATIONS[rep], "Algoritma": ix.ALGORITHMS[algo], "K": int(k) if algo != "dbscan" else None,
                     "Parameter": extra, **{c: m.get(c) for c in ["Silhouette", "DBI_MDS", "CHI_MDS", "Balance", "Cluster", "Noise", "Eta2_nol", "Waktu_s"]},
                     "Terbaik": key in results["best"].values(), "Kunci": key})
    return pd.DataFrame(rows)


def export_results(results: dict, out_dir: str | Path, zip_path: str | Path | None = None) -> Path:
    """Menulis folder ``hasil_model`` (JSON + CSV pendukung) dan opsional mengemasnya menjadi zip."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in results.items() if not k.startswith("_")}
    (out / "model_results.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps({**results["metadata"], "best": results["best"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    codes = results["codes"]
    weeks = [f"W{i}" for i in range(len(results["W"][0]))]
    pd.DataFrame(results["W"], columns=weeks).assign(Product_Code=codes)[["Product_Code", *weeks]].to_csv(out / "sales.csv", index=False)
    metrics_table(results).to_csv(out / "metrik_semua_model.csv", index=False)
    labels = pd.DataFrame({"Product_Code": codes})
    for rep, key in results["best"].items():
        if key:
            labels[f"{rep}__{key.split('|', 1)[1].replace('|', '_')}"] = results["runs"][key]["labels"]
    labels.to_csv(out / "label_model_terbaik.csv", index=False)
    if zip_path:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.iterdir()):
                archive.write(path, arcname=f"hasil_model/{path.name}")
        return Path(zip_path)
    return out


def load_results(path: str | Path) -> dict:
    """Memuat ``model_results.json`` (folder hasil_model atau path file langsung)."""
    path = Path(path)
    if path.is_dir():
        path = path / "model_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_to_result(run: dict, space: dict, linkage: str | None = None) -> SimpleNamespace:
    """Mengubah satu run tersimpan menjadi objek seperti ``ClusterResult`` untuk visualisasi & profil."""
    arr = lambda v: np.array([np.nan if x is None else x for x in v], dtype=float) if v is not None else None
    centers = np.array([[np.nan if x is None else x for x in row] for row in run["centers"]], dtype=float) if run["centers"] else np.empty((0, 0))
    return SimpleNamespace(
        labels=np.asarray(run["labels"], dtype=int), silhouette=arr(run["silhouette"]), dist_to_center=arr(run["dist_to_center"]),
        centers=centers, center_kind=run["center_kind"], medoids=np.asarray(run["medoids"], dtype=int),
        cluster_ids=np.asarray(run["cluster_ids"], dtype=int),
        center_xy=np.array(run["center_xy"], dtype=float).reshape(-1, 2) if run["center_xy"] else np.empty((0, 2)),
        center_dm=np.array(run["center_dm"], dtype=float) if run["center_dm"] else np.zeros((0, 0)),
        membership=np.array(run["membership"], dtype=float) if run["membership"] is not None else None,
        linkage=np.array(space["linkage"][linkage]["Z"], dtype=float) if linkage else None,
        eps=run["eps"], notes=run["notes"], metrics={k: (np.nan if v is None else v) for k, v in run["metrics"].items()},
        n_iter=run["n_iter"], runtime=run["metrics"].get("Waktu_s") or 0.0,
    )
