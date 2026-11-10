"""Mesin klasterisasi on-demand untuk dashboard (gaya R Shiny: pilih → jalankan → tampil).

Modul ini sengaja *tidak* bergantung pada Streamlit maupun numba sehingga bisa dipakai di
notebook, di dashboard, dan diuji terpisah. Semua perhitungan DTW memakai NumPy yang
divektorisasi di sepanjang pasangan deret (cepat untuk 811 × 52 tanpa kompilasi JIT).

Kombinasi yang didukung
-----------------------
Representasi : ``original`` (data asli), ``minmax`` (Min-Max per produk), ``zscore`` (scale/z-score per produk)
Jarak        : ``euclidean``, ``dtw`` (penuh), ``cdtw`` (pita Sakoe-Chiba), ``fastdtw`` (pakai numba bila ada)
Algoritma    : ``kmeans`` (rata-rata / DBA), ``kmedoids`` (PAM BUILD+SWAP), ``agglomerative``,
               ``dbscan`` (eps otomatis dari k-distance), ``fcm`` (Fuzzy C-Means / DBA), ``fcmdd`` (Fuzzy C-Medoids)

Definisi "centroid" yang ditampilkan
------------------------------------
* K-Means / FCM + Euclidean → rata-rata aritmetika anggota (pusat Lloyd).
* K-Means / FCM + DTW       → barycenter DBA (DTW Barycenter Averaging).
* K-Medoids / FCMdd         → medoid (produk nyata).
* Agglomerative / DBSCAN    → tidak punya pusat bawaan; ditampilkan medoid + rata-rata anggota.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples

REPRESENTATIONS = {
    "original": "Data asli",
    "minmax": "Min-Max per produk",
    "zscore": "Scale (z-score per produk)",
}
DISTANCES = {
    "euclidean": "Euclidean",
    "dtw": "DTW",
    "cdtw": "CDTW (Sakoe-Chiba)",
    "fastdtw": "FastDTW",
}
ALGORITHMS = {
    "kmeans": "K-Means (DBA untuk DTW)",
    "kmedoids": "K-Medoids PAM",
    "agglomerative": "Hierarchical (Agglomerative)",
    "dbscan": "DBSCAN",
    "fcm": "Fuzzy C-Means",
    "fcmdd": "Fuzzy C-Medoids",
}
CENTER_KIND = {"mean": "Rata-rata anggota", "dba": "Barycenter DBA", "medoid": "Medoid (produk nyata)"}


# =======================================================================================
# Representasi data
# =======================================================================================
def represent(W: np.ndarray, kind: str) -> np.ndarray:
    """Mengubah matriks penjualan (produk × minggu) ke representasi terpilih (row-wise)."""
    values = np.asarray(W, dtype=np.float64)
    if kind == "original":
        return values.copy()
    if kind == "minmax":
        low, high = values.min(axis=1, keepdims=True), values.max(axis=1, keepdims=True)
        span = high - low
        return np.where(span > 0, (values - low) / np.where(span > 0, span, 1.0), 0.0)
    if kind == "zscore":
        mean, std = values.mean(axis=1, keepdims=True), values.std(axis=1, keepdims=True)
        return np.where(std > 0, (values - mean) / np.where(std > 0, std, 1.0), 0.0)
    raise ValueError(f"Representasi tidak dikenal: {kind}")


# =======================================================================================
# DTW tervektorisasi
# =======================================================================================
def _band(i: int, m: int, window: int | None) -> tuple[int, int]:
    """Rentang kolom j (1-based, inklusif) yang boleh dikunjungi pada baris i."""
    if window is None:
        return 1, m
    return max(1, i - window), min(m, i + window)


def dtw_batch(A: np.ndarray, B: np.ndarray, window: int | None = None, return_cost: bool = False):
    """DTW untuk P pasangan sekaligus. A: (P, n), B: (P, m).

    Biaya lokal = selisih kuadrat; jarak akhir = sqrt(jumlah biaya pada jalur optimal),
    konsisten dengan definisi FastDTW di ``scripts/distance.py``.
    Jika ``return_cost`` True, juga mengembalikan matriks biaya kumulatif (P, n+1, m+1).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    P, n = A.shape
    m = B.shape[1]
    if window is not None:
        window = max(int(window), abs(n - m))
    if return_cost:
        D = np.full((P, n + 1, m + 1), np.inf)
        D[:, 0, 0] = 0.0
        for i in range(1, n + 1):
            lo, hi = _band(i, m, window)
            cost = (A[:, i - 1, None] - B[:, lo - 1:hi]) ** 2
            prev, cur = D[:, i - 1], D[:, i]
            for j in range(lo, hi + 1):
                cur[:, j] = cost[:, j - lo] + np.minimum(np.minimum(prev[:, j], prev[:, j - 1]), cur[:, j - 1])
        return np.sqrt(D[:, n, m]), D
    # Tata letak (kolom, pasangan) agar setiap langkah j membaca memori bersebelahan.
    At, Bt = np.ascontiguousarray(A.T), np.ascontiguousarray(B.T)
    prev = np.full((m + 1, P), np.inf)
    prev[0] = 0.0
    tmp = np.empty(P)
    for i in range(1, n + 1):
        cur = np.full((m + 1, P), np.inf)
        lo, hi = _band(i, m, window)
        cost = (Bt[lo - 1:hi] - At[i - 1]) ** 2
        for j in range(lo, hi + 1):
            np.minimum(prev[j], prev[j - 1], out=tmp)
            np.minimum(tmp, cur[j - 1], out=tmp)
            np.add(tmp, cost[j - lo], out=cur[j])
        prev = cur
    return np.sqrt(prev[m])


def dtw_paths(D: np.ndarray) -> list[np.ndarray]:
    """Backtracking jalur optimal dari matriks biaya kumulatif (P, n+1, m+1).

    Mengembalikan list berisi array (L, 2) indeks (i, j) 0-based untuk setiap pasangan.
    """
    P, n1, m1 = D.shape
    i = np.full(P, n1 - 1)
    j = np.full(P, m1 - 1)
    steps_i, steps_j, active_steps = [], [], []
    rows = np.arange(P)
    active = np.ones(P, dtype=bool)
    while active.any():
        steps_i.append(i - 1)
        steps_j.append(j - 1)
        active_steps.append(active.copy())
        diag = D[rows, np.maximum(i - 1, 0), np.maximum(j - 1, 0)]
        up = D[rows, np.maximum(i - 1, 0), j]
        left = D[rows, i, np.maximum(j - 1, 0)]
        choice = np.argmin(np.stack([diag, up, left]), axis=0)
        new_i = np.where(choice == 2, i, i - 1)
        new_j = np.where(choice == 1, j, j - 1)
        i = np.where(active, new_i, i)
        j = np.where(active, new_j, j)
        active = active & ~((i == 0) & (j == 0)) & (i > 0) & (j > 0)
    si, sj, sa = np.array(steps_i), np.array(steps_j), np.array(active_steps)
    return [np.c_[si[sa[:, p], p], sj[sa[:, p], p]][::-1] for p in range(P)]


def dtw_pair(a: np.ndarray, b: np.ndarray, window: int | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """DTW satu pasangan: (jarak, jalur (L,2), matriks biaya kumulatif tanpa baris/kolom 0)."""
    distance, D = dtw_batch(np.asarray(a)[None], np.asarray(b)[None], window, return_cost=True)
    return float(distance[0]), dtw_paths(D)[0], D[0, 1:, 1:]


def pairwise_dtw(X: np.ndarray, window: int | None = None, chunk: int = 40000, progress=None) -> np.ndarray:
    """Matriks jarak DTW/CDTW simetris untuk seluruh produk (diproses per blok pasangan)."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    rows, cols = np.triu_indices(n, k=1)
    values = np.empty(len(rows))
    for start in range(0, len(rows), chunk):
        stop = min(start + chunk, len(rows))
        values[start:stop] = dtw_batch(X[rows[start:stop]], X[cols[start:stop]], window)
        if progress:
            progress(stop / len(rows))
    D = np.zeros((n, n))
    D[rows, cols] = values
    D[cols, rows] = values
    return D


def cross_dtw(X: np.ndarray, centers: np.ndarray, window: int | None = None) -> np.ndarray:
    """Jarak DTW setiap deret ke setiap pusat: (n, k)."""
    n, k = len(X), len(centers)
    A = np.repeat(np.asarray(X, dtype=np.float64), k, axis=0)
    B = np.tile(np.asarray(centers, dtype=np.float64), (n, 1))
    return dtw_batch(A, B, window).reshape(n, k)


def dba_update(X: np.ndarray, center: np.ndarray, weights: np.ndarray | None = None,
               window: int | None = None, iterations: int = 3) -> np.ndarray:
    """DTW Barycenter Averaging berbobot (Petitjean dkk., 2011), divektorisasi per iterasi."""
    X = np.asarray(X, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).copy()
    w = np.ones(len(X)) if weights is None else np.asarray(weights, dtype=np.float64)
    keep = w > 1e-10
    X, w = X[keep], w[keep]
    if len(X) == 0:
        return center
    for _ in range(int(iterations)):
        _, D = dtw_batch(np.repeat(center[None], len(X), axis=0), X, window, return_cost=True)
        sums = np.zeros_like(center)
        counts = np.zeros_like(center)
        for p, path in enumerate(dtw_paths(D)):
            np.add.at(sums, path[:, 0], w[p] * X[p, path[:, 1]])
            np.add.at(counts, path[:, 0], w[p])
        valid = counts > 1e-12
        center[valid] = sums[valid] / counts[valid]
    return center


def distance_matrix(X: np.ndarray, metric: str, window: int = 5, radius: int = 2,
                    cache_dir: Path | None = None, progress=None) -> tuple[np.ndarray, str]:
    """Matriks jarak penuh + catatan. Disimpan ke cache disk bila ``cache_dir`` diberikan."""
    X = np.asarray(X, dtype=np.float64)
    tag = {"euclidean": "", "dtw": "", "cdtw": f"_w{window}", "fastdtw": f"_r{radius}"}[metric]
    digest = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()[:16]
    path = Path(cache_dir) / f"interactive_{metric}{tag}_{digest}.npy" if cache_dir else None
    if path is not None and path.is_file():
        return np.load(path), "dimuat dari cache"
    note = ""
    if metric == "euclidean":
        D = squareform(pdist(X, metric="euclidean"))
    elif metric == "dtw":
        D = pairwise_dtw(X, None, progress=progress)
    elif metric == "cdtw":
        D = pairwise_dtw(X, int(window), progress=progress)
    elif metric == "fastdtw":
        legacy = Path(cache_dir) / f"fastdtw_{hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()[:16]}_r{radius}.npy" if cache_dir else None
        if legacy is not None and legacy.is_file():
            D, note = np.load(legacy), "memakai matriks FastDTW notebook"
        else:
            try:
                import numba
                from scripts.distance import compute_distance_matrix  # numba
                threads = int(getattr(numba.config, "NUMBA_NUM_THREADS", 1))  # Colab/Cloud hanya 2 core
                D, note = compute_distance_matrix(X, radius=radius, n_threads=threads), "FastDTW numba"
            except Exception:  # numba tidak tersedia → aproksimasi CDTW dengan pita sebesar radius
                D = pairwise_dtw(X, max(1, int(radius)), progress=progress)
                note = f"numba tidak tersedia; FastDTW diaproksimasi CDTW pita {radius}"
    else:
        raise ValueError(f"Jarak tidak dikenal: {metric}")
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, D)
        except OSError:
            pass
    return D, note or "dihitung"


# =======================================================================================
# Proyeksi 2D (MDS klasik) untuk visual fviz_cluster
# =======================================================================================
@dataclass
class Embedding:
    """Hasil MDS klasik: koordinat, nilai eigen, dan bahan proyeksi titik baru."""
    coords: np.ndarray            # (n, d)
    explained: np.ndarray         # (d,) rasio varians
    eigvals: np.ndarray           # (d,)
    eigvecs: np.ndarray           # (n, d)
    col_mean_sq: np.ndarray       # (n,) rerata kolom D²

    def project(self, distances_to_points: np.ndarray) -> np.ndarray:
        """Proyeksi titik baru (mis. barycenter DBA) dengan formula Gower dari jaraknya ke semua titik."""
        d2 = np.asarray(distances_to_points, dtype=float) ** 2
        b = -0.5 * (d2 - self.col_mean_sq[None, :] if d2.ndim == 2 else d2 - self.col_mean_sq)
        return (b @ self.eigvecs) / np.sqrt(self.eigvals)


def classical_mds(D: np.ndarray, dims: int = 5) -> Embedding:
    """MDS klasik (Torgerson). Untuk jarak Euclidean hasilnya identik dengan PCA."""
    D2 = np.asarray(D, dtype=float) ** 2
    n = len(D2)
    J = np.eye(n) - 1.0 / n
    Bm = -0.5 * J @ D2 @ J
    values, vectors = np.linalg.eigh(Bm)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    positive = values[values > 1e-9]
    keep = min(dims, len(positive))
    lam, vec = values[:keep], vectors[:, :keep]
    return Embedding(
        coords=vec * np.sqrt(lam), explained=lam / positive.sum(), eigvals=lam, eigvecs=vec,
        col_mean_sq=D2.mean(axis=0) - D2.mean(),
    )


# =======================================================================================
# Algoritma
# =======================================================================================
def _init_pp(D: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Inisialisasi k-means++ berbasis jarak (sama dengan ``scripts/clustering.py``)."""
    rng = np.random.default_rng(seed)
    n = len(D)
    chosen = [int(rng.integers(n))]
    while len(chosen) < k:
        weights = np.min(D[:, chosen], axis=1) ** 2
        weights[chosen] = 0.0
        chosen.append(int(rng.choice(n, p=weights / weights.sum())) if weights.sum() > 0
                      else int(rng.choice(np.setdiff1d(np.arange(n), chosen))))
    return np.asarray(chosen, dtype=int)


def _memberships(dist: np.ndarray, m: float) -> np.ndarray:
    """Keanggotaan fuzzy dari jarak ke pusat (u_ij ∝ d_ij^(-2/(m-1)))."""
    d = np.maximum(np.asarray(dist, dtype=float), 1e-12)
    inv = d ** (-2.0 / (m - 1.0))
    return inv / inv.sum(axis=1, keepdims=True)


def pam(D: np.ndarray, k: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, int]:
    """K-Medoids PAM klasik: BUILD rakus + SWAP terbaik per iterasi (tervektorisasi)."""
    n = len(D)
    medoids = [int(np.argmin(D.sum(axis=1)))]
    for _ in range(1, k):
        nearest = D[:, medoids].min(axis=1)
        gain = np.maximum(nearest[:, None] - D, 0).sum(axis=0)
        gain[medoids] = -np.inf
        medoids.append(int(np.argmax(gain)))
    medoids = np.asarray(medoids)
    iteration = 0
    for iteration in range(1, max_iter + 1):
        dm = D[:, medoids]
        order = np.argsort(dm, axis=1)
        assign = order[:, 0]
        d1 = dm[np.arange(n), assign]
        d2 = dm[np.arange(n), order[:, 1]] if k > 1 else np.full(n, np.inf)
        base = np.minimum(D - d1[:, None], 0).sum(axis=0)        # (h,) bagian dari titik di luar cluster i
        best_delta, best = 0.0, None
        for i in range(k):
            mask = assign == i
            Di = D[mask]                                          # (|Ci|, n)
            own = (np.minimum(Di, d2[mask, None]) - d1[mask, None]).sum(axis=0)
            correction = np.minimum(Di - d1[mask, None], 0).sum(axis=0)
            delta = base - correction + own
            delta[medoids] = np.inf
            h = int(np.argmin(delta))
            if delta[h] < best_delta - 1e-10:
                best_delta, best = float(delta[h]), (i, h)
        if best is None:
            break
        medoids[best[0]] = best[1]
    labels = np.argmin(D[:, medoids], axis=1)
    return labels, medoids, iteration


def _kmeans_dtw(X, D, k, window, seed, n_init, max_iter=15, dba_iter=3):
    """K-Means dengan barycenter DBA (jarak DTW/CDTW)."""
    best = None
    for run in range(max(1, n_init)):
        centers = X[_init_pp(D, k, seed + run)].copy()
        previous = None
        for it in range(max_iter):
            dist = cross_dtw(X, centers, window)
            labels = dist.argmin(axis=1)
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels
            for c in range(k):
                members = labels == c
                if members.any():
                    centers[c] = dba_update(X[members], centers[c], None, window, dba_iter)
                else:
                    centers[c] = X[dist.min(axis=1).argmax()]
        dist = cross_dtw(X, centers, window)
        labels = dist.argmin(axis=1)
        objective = float((dist.min(axis=1) ** 2).sum())
        if best is None or objective < best[0]:
            best = (objective, labels, centers.copy(), dist, it + 1)
    return best


def _fcm_dtw(X, D, k, m, window, seed, max_iter=8, dba_iter=2, tol=1e-4):
    """Fuzzy C-Means dengan pusat DBA berbobot u^m."""
    centers = X[_init_pp(D, k, seed)].copy()
    it = 0
    for it in range(1, max_iter + 1):
        U = _memberships(cross_dtw(X, centers, window), m)
        updated = np.stack([dba_update(X, centers[c], U[:, c] ** m, window, dba_iter) for c in range(k)])
        shift = float(np.abs(updated - centers).max())
        centers = updated
        if shift < tol:
            break
    dist = cross_dtw(X, centers, window)
    return _memberships(dist, m), centers, dist, it


def _fcm_euclid(X, k, m, seed, max_iter=150, tol=1e-6):
    """Fuzzy C-Means klasik (Bezdek) pada ruang Euclidean."""
    rng = np.random.default_rng(seed)
    U = rng.dirichlet(np.ones(k), size=len(X))
    it = 0
    for it in range(1, max_iter + 1):
        Um = U ** m
        centers = (Um.T @ X) / Um.sum(axis=0)[:, None]
        dist = np.sqrt(((X[:, None, :] - centers[None]) ** 2).sum(axis=2))
        U_new = _memberships(dist, m)
        if np.abs(U_new - U).max() < tol:
            U = U_new
            break
        U = U_new
    return U, centers, dist, it


def _fcmdd(D, k, m, seed, max_iter=30):
    """Fuzzy C-Medoids (Krishnapuram dkk.) dengan medoid berbobot keanggotaan."""
    medoids = _init_pp(D, k, seed)
    D2 = D ** 2
    it = 0
    for it in range(1, max_iter + 1):
        U = _memberships(D[:, medoids], m)
        scores = D2.T @ (U ** m)
        proposed = []
        for c in range(k):
            for cand in np.argsort(scores[:, c]):
                if int(cand) not in proposed:
                    proposed.append(int(cand))
                    break
        proposed = np.asarray(proposed)
        if np.array_equal(proposed, medoids):
            break
        medoids = proposed
    return _memberships(D[:, medoids], m), medoids, it


def kdistance(D: np.ndarray, k: int) -> np.ndarray:
    """Jarak ke tetangga ke-k setiap titik, terurut naik (untuk plot kNNdist gaya R dbscan)."""
    return np.sort(np.sort(D, axis=1)[:, int(k)])


@dataclass
class ClusterResult:
    """Semua keluaran satu kali jalankan, siap untuk visual dan tabel."""
    params: dict
    labels: np.ndarray
    centers: np.ndarray                 # (k, T) pusat pada ruang representasi
    center_kind: str                    # mean | dba | medoid
    medoids: np.ndarray                 # indeks medoid per cluster (urut cluster)
    cluster_ids: np.ndarray             # ID cluster non-noise terurut
    dist_to_center: np.ndarray          # (n,) jarak produk ke pusat cluster-nya (NaN untuk noise)
    center_dist: np.ndarray             # (n, k) jarak tiap produk ke tiap pusat
    membership: np.ndarray | None = None
    linkage: np.ndarray | None = None
    eps: float | None = None
    n_iter: int = 0
    runtime: float = 0.0
    notes: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    silhouette: np.ndarray | None = None
    center_xy: np.ndarray | None = None  # (k, 2) posisi pusat di peta MDS


def _medoids_of(D: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> np.ndarray:
    out = []
    for c in ids:
        members = np.flatnonzero(labels == c)
        out.append(int(members[np.argmin(D[np.ix_(members, members)].sum(axis=1))]))
    return np.asarray(out, dtype=int)


def run_clustering(
    X: np.ndarray,
    D: np.ndarray,
    algorithm: str,
    metric: str,
    k: int = 2,
    linkage_method: str = "average",
    min_samples: int = 7,
    eps_percentile: float = 85.0,
    eps: float | None = None,
    fuzziness: float = 2.0,
    window: int = 5,
    seed: int = 42,
    n_init: int = 3,
) -> ClusterResult:
    """Menjalankan satu algoritma pada representasi X dan matriks jarak D."""
    start = time.perf_counter()
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    euclid = metric == "euclidean"
    win = None if metric in ("dtw", "fastdtw") else (int(window) if metric == "cdtw" else None)
    notes: list[str] = []
    membership = linkage_matrix = used_eps = None
    n_iter = 0
    if algorithm == "kmeans":
        if euclid:
            model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
            labels, centers, n_iter = model.labels_, model.cluster_centers_, int(model.n_iter_)
            center_dist = np.sqrt(((X[:, None, :] - centers[None]) ** 2).sum(axis=2))
            kind = "mean"
        else:
            if metric == "fastdtw":
                notes.append("Pembaruan pusat DBA memakai DTW penuh (FastDTW hanya untuk matriks jarak).")
            _, labels, centers, center_dist, n_iter = _kmeans_dtw(X, D, k, win, seed, n_init)
            kind = "dba"
    elif algorithm == "kmedoids":
        labels, medoids, n_iter = pam(D, k)
        centers, kind, center_dist = X[medoids], "medoid", D[:, medoids]
    elif algorithm == "agglomerative":
        if linkage_method == "ward" and not euclid:
            notes.append("Ward hanya sah untuk jarak Euclidean; dipakai 'average'.")
            linkage_method = "average"
        linkage_matrix = linkage(squareform(D, checks=False), method=linkage_method)
        raw = fcluster(linkage_matrix, t=k, criterion="maxclust")
        _, labels = np.unique(raw, return_inverse=True)
        kind = "medoid"
    elif algorithm == "dbscan":
        used_eps = float(eps) if eps else float(np.percentile(np.sort(D, axis=1)[:, int(min_samples)], eps_percentile))
        labels = DBSCAN(eps=used_eps, min_samples=int(min_samples), metric="precomputed").fit_predict(D)
        kind = "medoid"
        notes.append(f"eps = {used_eps:.4g} (persentil ke-{eps_percentile:g} jarak tetangga ke-{min_samples})" if not eps else f"eps manual = {used_eps:.4g}")
    elif algorithm == "fcm":
        if euclid:
            membership, centers, center_dist, n_iter = _fcm_euclid(X, k, fuzziness, seed)
            kind = "mean"
        else:
            membership, centers, center_dist, n_iter = _fcm_dtw(X, D, k, fuzziness, win, seed)
            kind = "dba"
        labels = membership.argmax(axis=1)
    elif algorithm == "fcmdd":
        membership, medoids, n_iter = _fcmdd(D, k, fuzziness, seed)
        labels = membership.argmax(axis=1)
        centers, kind, center_dist = X[medoids], "medoid", D[:, medoids]
    else:
        raise ValueError(f"Algoritma tidak dikenal: {algorithm}")

    labels = np.asarray(labels, dtype=int)
    ids = np.array(sorted(set(labels.tolist()) - {-1}), dtype=int)
    if algorithm in ("agglomerative", "dbscan"):
        medoids = _medoids_of(D, labels, ids) if len(ids) else np.array([], dtype=int)
        centers = X[medoids] if len(ids) else np.empty((0, X.shape[1]))
        center_dist = D[:, medoids] if len(ids) else np.empty((n, 0))
    else:
        if algorithm in ("kmeans", "fcm"):
            # cluster bisa kosong → buang pusat yatim agar indeks konsisten
            used = np.isin(np.arange(len(centers)), ids)
            remap = -np.ones(len(centers), dtype=int)
            remap[used] = np.arange(used.sum())
            centers, center_dist = centers[used], center_dist[:, used]
            if membership is not None:
                membership = membership[:, used] / membership[:, used].sum(axis=1, keepdims=True)
            labels = np.where(labels >= 0, remap[np.maximum(labels, 0)], -1)
            ids = np.arange(used.sum())
            medoids = _medoids_of(D, labels, ids)  # produk representatif (terdekat ke semua anggota)
    own = np.full(n, np.nan)
    for pos, c in enumerate(ids):
        mask = labels == c
        own[mask] = center_dist[mask, pos]
    return ClusterResult(
        params=dict(algorithm=algorithm, metric=metric, k=k, linkage=linkage_method, min_samples=min_samples,
                    eps_percentile=eps_percentile, fuzziness=fuzziness, window=window, seed=seed),
        labels=labels, centers=np.asarray(centers), center_kind=kind, medoids=medoids, cluster_ids=ids,
        dist_to_center=own, center_dist=np.asarray(center_dist), membership=membership, linkage=linkage_matrix,
        eps=used_eps, n_iter=int(n_iter), runtime=time.perf_counter() - start, notes=notes,
    )


# =======================================================================================
# Evaluasi
# =======================================================================================
def medoid_dbi(D: np.ndarray, labels: np.ndarray) -> float:
    """Davies–Bouldin adaptasi berbasis medoid pada matriks jarak apa pun."""
    ids = sorted(set(labels.tolist()) - {-1})
    if len(ids) < 2:
        return float("nan")
    med = _medoids_of(D, labels, np.asarray(ids))
    scatter = np.array([D[labels == c, m].mean() for c, m in zip(ids, med)])
    between = D[np.ix_(med, med)].copy()
    np.fill_diagonal(between, np.inf)
    ratio = (scatter[:, None] + scatter[None, :]) / np.maximum(between, 1e-12)
    np.fill_diagonal(ratio, -np.inf)
    return float(ratio.max(axis=1).mean())


def balance_score(labels: np.ndarray) -> float:
    """Entropi ukuran cluster ternormalisasi: 1 = ukuran seimbang, 0 = satu cluster dominan."""
    ids, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(ids) < 2:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(ids)))


def evaluate(result: ClusterResult, D: np.ndarray, embedding: Embedding, W: np.ndarray) -> ClusterResult:
    """Mengisi metrik kualitas, Silhouette per produk, dan posisi pusat di peta MDS."""
    y = result.labels
    covered = y != -1
    ids = result.cluster_ids
    metrics = {
        "Silhouette": np.nan, "DBI_medoid": np.nan, "DBI_MDS": np.nan, "CHI_MDS": np.nan,
        "Balance": balance_score(y), "Coverage": float(covered.mean()), "Noise": int((~covered).sum()),
        "Cluster": int(len(ids)), "Eta2_nol": np.nan, "Waktu_s": result.runtime,
        "Within": float(np.nansum(result.dist_to_center)),
    }
    zeros = (np.asarray(W)[covered] == 0).sum(axis=1)
    if covered.any() and zeros.var() > 0:
        groups = y[covered]
        between = sum((groups == c).sum() * (zeros[groups == c].mean() - zeros.mean()) ** 2 for c in ids)
        metrics["Eta2_nol"] = float(between / ((zeros - zeros.mean()) ** 2).sum())
    silhouette = np.full(len(y), np.nan)
    if 2 <= len(ids) < covered.sum():
        keep = np.flatnonzero(covered)
        silhouette[keep] = silhouette_samples(D[np.ix_(keep, keep)], y[keep], metric="precomputed")
        metrics["Silhouette"] = float(np.nanmean(silhouette))
        metrics["DBI_medoid"] = medoid_dbi(D, y)
        emb = embedding.coords[keep]
        metrics["DBI_MDS"] = float(davies_bouldin_score(emb, y[keep]))
        metrics["CHI_MDS"] = float(calinski_harabasz_score(emb, y[keep]))
    if result.membership is not None:
        U = result.membership
        metrics["FPC"] = float((U ** 2).sum() / len(U))
        metrics["Entropi_partisi"] = float(-(U * np.log(np.maximum(U, 1e-15))).sum() / len(U))
    if result.center_kind == "medoid" or len(ids) == 0:
        xy = embedding.coords[result.medoids, :2] if len(ids) else np.empty((0, 2))
    else:
        xy = embedding.project(result.center_dist.T)[:, :2]
    result.metrics, result.silhouette, result.center_xy = metrics, silhouette, xy
    return result


def nbclust(X, D, embedding, W, algorithm, metric, k_values=range(2, 9), **kwargs) -> list[dict]:
    """Kurva siku (within) dan rata-rata Silhouette untuk beberapa k (gaya fviz_nbclust)."""
    rows = []
    for k in k_values:
        res = evaluate(run_clustering(X, D, algorithm, metric, k=int(k), **kwargs), D, embedding, W)
        rows.append({"k": int(k), **res.metrics})
    return rows


def rank_methods(table: list[dict]) -> list[dict]:
    """Peringkat rata-rata Silhouette↑, DBI(MDS)↓, CHI(MDS)↑ dengan bobot sama (seperti app referensi)."""
    import pandas as pd
    frame = pd.DataFrame(table)
    frame["Rank_Silhouette"] = frame["Silhouette"].rank(ascending=False, method="min")
    frame["Rank_DBI"] = frame["DBI_MDS"].rank(ascending=True, method="min")
    frame["Rank_CHI"] = frame["CHI_MDS"].rank(ascending=False, method="min")
    frame["Skor_Ranking"] = frame[["Rank_Silhouette", "Rank_DBI", "Rank_CHI"]].mean(axis=1)
    return frame.sort_values(["Skor_Ranking", "Silhouette"], ascending=[True, False]).to_dict("records")


def cluster_profile_rows(result: ClusterResult, W: np.ndarray, codes: np.ndarray) -> list[dict]:
    """Profil bisnis per cluster (data asli) + interpretasi tren 13 minggu awal vs akhir."""
    W = np.asarray(W, dtype=float)
    total = W.sum()
    rows = []
    groups = list(result.cluster_ids) + ([-1] if (result.labels == -1).any() else [])
    for c in groups:
        mask = result.labels == c
        G = W[mask]
        profile = G.mean(axis=0)
        first, last = profile[:13].mean(), profile[-13:].mean()
        change = 100 * (last - first) / first if first > 0 else np.nan
        cv = float(profile.std() / profile.mean()) if profile.mean() > 0 else np.nan
        zero_median = float(np.median((G == 0).sum(axis=1)))
        if c == -1:
            trend, action = "Noise", "Tinjau manual; pola tidak cukup dekat dengan cluster mana pun."
        elif np.isnan(change) or abs(change) < 5:
            trend, action = "Stabil", "Pertahankan kebijakan stok; pantau deviasi mingguan."
        elif change < 0:
            trend, action = "Tren melemah", "Evaluasi reorder point dan hindari penambahan stok sebelum penurunan terkonfirmasi."
        else:
            trend, action = "Tren menguat", "Tinjau kenaikan kebutuhan dan pertimbangkan penyesuaian safety stock secara bertahap."
        if zero_median >= 30 and c != -1:
            persona = "Sporadis"
        elif zero_median <= 4 and c != -1:
            persona = "Aktif & konsisten"
        elif c != -1:
            persona = "Fluktuatif"
        else:
            persona = "Tak terkelompok"
        pos = int(np.flatnonzero(result.cluster_ids == c)[0]) if c != -1 else None
        rows.append({
            "Cluster": int(c), "Persona": persona, "Tren": trend, "Rekomendasi": action,
            "Jumlah_Produk": int(mask.sum()), "Proporsi_%": 100 * float(mask.mean()),
            "Total_Penjualan": float(G.sum()), "Porsi_Volume_%": 100 * float(G.sum() / total) if total else 0.0,
            "Rata2_Mingguan": float(G.mean()), "Median_Minggu_Nol": zero_median, "CV_Profil": cv,
            "Perubahan_13mg_%": change,
            "Produk_Representatif": str(codes[result.medoids[pos]]) if pos is not None else "–",
            "Rata2_Jarak_ke_Pusat": float(np.nanmean(result.dist_to_center[mask])) if c != -1 else np.nan,
        })
    return rows
