"""Fungsi pemuatan, validasi, dan ringkasan data penjualan mingguan."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def validate_sales_data(
    data: pd.DataFrame,
    id_column: str = "Product_Code",
    week_columns: list[str] | None = None,
    allow_missing: bool = False,
    allow_negative: bool = False,
) -> pd.DataFrame:
    """Validasi struktur, ID, dan nilai penjualan; mengembalikan salinan yang telah dirapikan."""
    if week_columns is None:
        week_columns = [f"W{i}" for i in range(52)]
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Data penjualan harus berupa DataFrame yang tidak kosong.")

    frame = data.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    required = [id_column, *week_columns]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing_columns}")

    if frame[id_column].isna().any():
        raise ValueError(f"Kolom {id_column} mengandung ID kosong.")
    frame[id_column] = frame[id_column].astype(str).str.strip()
    if frame[id_column].eq("").any() or frame[id_column].duplicated().any():
        raise ValueError(f"{id_column} harus terisi dan unik untuk setiap produk.")

    for column in week_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame[week_columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Data penjualan mengandung nilai tak hingga.")
    if not allow_missing and np.isnan(values).any():
        locations = np.argwhere(np.isnan(values))[:10].tolist()
        raise ValueError(f"Data penjualan mengandung nilai kosong/teks. Contoh posisi: {locations}")
    if not allow_negative and np.nanmin(values) < 0:
        raise ValueError("Penjualan negatif ditemukan. Periksa apakah nilai tersebut retur/koreksi.")
    if allow_missing and np.isnan(values).any():
        medians = frame[week_columns].median(axis=0)
        frame[week_columns] = frame[week_columns].fillna(medians).fillna(0.0)
    return frame[[id_column, *week_columns]].reset_index(drop=True)


def load_sales_data(
    file_path: str | Path,
    id_column: str = "Product_Code",
    week_columns: list[str] | None = None,
    delimiter: str = ",",
    allow_missing: bool = False,
    allow_negative: bool = False,
) -> pd.DataFrame:
    """Membaca CSV penjualan dan menjalankan validasi skema serta nilai."""
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {path.resolve()}")
    frame = pd.read_csv(path, sep=delimiter, dtype={id_column: "string"})
    return validate_sales_data(
        frame, id_column=id_column, week_columns=week_columns,
        allow_missing=allow_missing, allow_negative=allow_negative,
    )


def build_product_statistics(
    data: pd.DataFrame,
    id_column: str = "Product_Code",
    week_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Membentuk statistik level produk dari penjualan asli selama seluruh minggu."""
    if week_columns is None:
        week_columns = [f"W{i}" for i in range(52)]
    values = data[week_columns].to_numpy(dtype=float)
    totals = values.sum(axis=1)
    active_count = (values > 0).sum(axis=1)
    peak_index = values.argmax(axis=1)
    return pd.DataFrame({
        "Produk": data[id_column].astype(str).to_numpy(),
        "Total volume": totals,
        "Rata-rata / minggu": values.mean(axis=1),
        "Minggu nol": (values == 0).sum(axis=1),
        "Minggu aktif (%)": 100.0 * active_count / values.shape[1],
        "Puncak mingguan": values.max(axis=1),
        "Minggu puncak": [f"W{i}" for i in peak_index],
    })


def build_weekly_statistics(
    data: pd.DataFrame,
    week_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Meringkas volume total, median, dan persentase produk aktif pada setiap minggu."""
    if week_columns is None:
        week_columns = [f"W{i}" for i in range(52)]
    values = data[week_columns].to_numpy(dtype=float)
    return pd.DataFrame({
        "Minggu": np.arange(values.shape[1]),
        "Total_Penjualan": values.sum(axis=0),
        "Median_Penjualan_Produk": np.median(values, axis=0),
        "Persen_Produk_Aktif": 100.0 * (values > 0).mean(axis=0),
    })
