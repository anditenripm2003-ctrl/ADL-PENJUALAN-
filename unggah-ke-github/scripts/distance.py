"""Implementasi FastDTW multiresolusi terakselerasi Numba dan utilitas matriks jarak."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from numba import njit, prange, set_num_threads


@njit(cache=True)
def fastdtw_numba(a: np.ndarray, b: np.ndarray, radius: int = 2, return_path: bool = False):
    """Menghitung FastDTW univariat dengan biaya lokal kuadrat dan akar biaya akhir."""
    n0 = len(a)
    if n0 != len(b) or n0 == 0 or radius < 1:
        return np.inf, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    max_levels = 16
    aa = np.zeros((max_levels, n0), dtype=np.float64)
    bb = np.zeros((max_levels, n0), dtype=np.float64)
    lengths = np.zeros(max_levels, dtype=np.int32)
    aa[0, :] = a
    bb[0, :] = b
    lengths[0] = n0
    nlevels = 1

    while lengths[nlevels - 1] >= radius + 2 and nlevels < max_levels:
        m = lengths[nlevels - 1] // 2
        if m < 1:
            break
        for q in range(m):
            aa[nlevels, q] = (aa[nlevels - 1, 2 * q] + aa[nlevels - 1, 2 * q + 1]) / 2.0
            bb[nlevels, q] = (bb[nlevels - 1, 2 * q] + bb[nlevels - 1, 2 * q + 1]) / 2.0
        lengths[nlevels] = m
        nlevels += 1

    path_i = np.zeros(2 * n0 + 2, dtype=np.int32)
    path_j = np.zeros(2 * n0 + 2, dtype=np.int32)
    path_len = 0
    final_cost = np.inf

    for level in range(nlevels - 1, -1, -1):
        n = lengths[level]
        mask = np.zeros((n, n), dtype=np.bool_)
        if level == nlevels - 1:
            mask[:, :] = True
        else:
            for t in range(path_len):
                base_i = path_i[t]
                base_j = path_j[t]
                for di in range(-radius, radius + 1):
                    ii = base_i + di
                    if ii < 0 or ii >= lengths[level + 1]:
                        continue
                    for dj in range(-radius, radius + 1):
                        jj = base_j + dj
                        if jj < 0 or jj >= lengths[level + 1]:
                            continue
                        for ai in range(2):
                            x = ii * 2 + ai
                            if x >= n:
                                continue
                            for bj in range(2):
                                y = jj * 2 + bj
                                if y < n:
                                    mask[x, y] = True
            # Jaga agar titik awal dan akhir selalu berada dalam search window.
            mask[0, 0] = True
            mask[n - 1, n - 1] = True

        dp = np.full((n + 1, n + 1), np.inf, dtype=np.float64)
        move = np.zeros((n + 1, n + 1), dtype=np.int8)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if not mask[i - 1, j - 1]:
                    continue
                delta = aa[level, i - 1] - bb[level, j - 1]
                local_cost = delta * delta
                best = dp[i - 1, j]
                direction = 1
                if dp[i, j - 1] < best:
                    best = dp[i, j - 1]
                    direction = 2
                if dp[i - 1, j - 1] < best:
                    best = dp[i - 1, j - 1]
                    direction = 3
                dp[i, j] = best + local_cost
                move[i, j] = direction

        final_cost = dp[n, n]
        if not np.isfinite(final_cost):
            return np.inf, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        i, j = n, n
        p = 0
        while i > 0 and j > 0:
            path_i[p] = i - 1
            path_j[p] = j - 1
            p += 1
            direction = move[i, j]
            if direction == 1:
                i -= 1
            elif direction == 2:
                j -= 1
            elif direction == 3:
                i -= 1
                j -= 1
            else:
                return np.inf, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        # Jalur disimpan dari awal ke akhir.
        for t in range(p // 2):
            tmp_i = path_i[t]
            path_i[t] = path_i[p - 1 - t]
            path_i[p - 1 - t] = tmp_i
            tmp_j = path_j[t]
            path_j[t] = path_j[p - 1 - t]
            path_j[p - 1 - t] = tmp_j
        path_len = p

    distance = np.sqrt(final_cost)
    if return_path:
        return distance, path_i[:path_len].copy(), path_j[:path_len].copy()
    return distance, path_i[:0], path_j[:0]


@njit(parallel=True, cache=True)
def _all_pairs_fastdtw(X: np.ndarray, radius: int):
    """Kernel paralel untuk menghitung seluruh pasangan jarak antarpola."""
    n = X.shape[0]
    distances = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(i + 1, n):
            distance, _, _ = fastdtw_numba(X[i], X[j], radius, False)
            distances[i, j] = distance
            distances[j, i] = distance
    return distances


@njit(parallel=True, cache=True)
def _cross_fastdtw(X: np.ndarray, centers: np.ndarray, radius: int):
    """Kernel paralel untuk jarak setiap seri pada X ke setiap prototype."""
    output = np.zeros((len(X), len(centers)), dtype=np.float64)
    for i in prange(len(X)):
        for j in range(len(centers)):
            output[i, j] = fastdtw_numba(X[i], centers[j], radius, False)[0]
    return output


def exact_dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Menghitung DTW eksak sebagai pemeriksaan pembanding pada seri pendek."""
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("Seri DTW tidak boleh kosong.")
    dp = np.full((len(x) + 1, len(y) + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            local = (x[i - 1] - y[j - 1]) ** 2
            dp[i, j] = local + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(np.sqrt(dp[-1, -1]))


def compute_distance_matrix(X: np.ndarray, radius: int = 2, n_threads: int = 4) -> np.ndarray:
    """Menghitung matriks jarak FastDTW simetris untuk semua produk."""
    values = np.ascontiguousarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("X harus berupa matriks produk × minggu dengan minimal dua baris dan kolom.")
    if not np.isfinite(values).all():
        raise ValueError("X mengandung NaN atau inf.")
    if radius < 1:
        raise ValueError("radius FastDTW minimal 1.")
    set_num_threads(max(1, int(n_threads)))
    distances = _all_pairs_fastdtw(values, int(radius))
    np.fill_diagonal(distances, 0.0)
    if not np.isfinite(distances).all() or (distances < 0).any() or not np.allclose(distances, distances.T):
        raise RuntimeError("Matriks jarak FastDTW tidak valid.")
    return distances


def compute_cross_distance(X: np.ndarray, centers: np.ndarray, radius: int = 2) -> np.ndarray:
    """Menghitung jarak FastDTW antara kumpulan seri dan kumpulan prototype."""
    values = np.ascontiguousarray(X, dtype=np.float64)
    prototypes = np.ascontiguousarray(centers, dtype=np.float64)
    if values.ndim != 2 or prototypes.ndim != 2 or values.shape[1] != prototypes.shape[1]:
        raise ValueError("X dan centers harus memiliki jumlah minggu yang sama.")
    return _cross_fastdtw(values, prototypes, int(radius))


def data_digest(X: np.ndarray) -> str:
    """Menghasilkan sidik data singkat untuk validasi cache matriks jarak."""
    values = np.ascontiguousarray(X, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def load_or_compute_distance_matrix(
    X: np.ndarray,
    radius: int,
    cache_dir: str | Path,
    reuse_cache: bool = True,
    n_threads: int = 4,
) -> tuple[np.ndarray, Path, bool]:
    """Memuat cache matriks FastDTW yang cocok atau menghitung dan menyimpannya."""
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    digest = data_digest(X)
    cache_path = folder / f"fastdtw_{digest}_r{int(radius)}.npy"
    if reuse_cache and cache_path.exists():
        cached = np.load(cache_path, mmap_mode=None)
        expected = (len(X), len(X))
        valid = cached.shape == expected and np.isfinite(cached).all()
        valid = valid and np.allclose(cached, cached.T) and np.allclose(np.diag(cached), 0.0)
        if valid:
            return cached, cache_path, True
        cache_path.unlink()
    distances = compute_distance_matrix(X, radius=radius, n_threads=n_threads)
    np.save(cache_path, distances)
    return distances, cache_path, False
