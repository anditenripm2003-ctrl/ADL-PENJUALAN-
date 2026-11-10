# Ruang Sales · Klasterisasi Pola Penjualan (FastDTW)

Dua tahap, dua notebook Colab:

| Tahap | Notebook | Hasil |
|---|---|---|
| 1. Bangun model | `notebook/01_bangun_model_colab.ipynb` | `hasil_model.zip` (3 representasi × FastDTW × 6 algoritma, 105 model, model terbaik per representasi) |
| 2. Dashboard | `notebook/02_jalankan_streamlit_colab.ipynb` | Link dashboard Streamlit `https://….trycloudflare.com` |

## Langkah A–Z di Google Colab

1. Buka colab.research.google.com → **File → Upload notebook** → `01_bangun_model_colab.ipynb`.
2. Seret `Sales transaction (1).csv` ke panel 📁 **Files** (pemisah `;` atau `,` otomatis).
3. **Runtime → Run all**. Tunggu ±5–10 menit (Sesi 4 paling lama). `hasil_model.zip` otomatis terunduh.
4. Di Colab yang **sama** (jangan tutup sesi): **File → Upload notebook** → `02_jalankan_streamlit_colab.ipynb`.
5. **Runtime → Run all** (±1 menit) → klik link 🌐 di sel terakhir.

Mengubah pengaturan (K, linkage, eps DBSCAN, guardrail) cukup di sel `%%writefile config.py` notebook 01, lalu Run all lagi.

## Deploy permanen (opsional)

Taruh isi `hasil_model.zip` di folder `hasil_model/` proyek ini, lalu deploy `app.py` ke Streamlit Community Cloud / Hugging Face Spaces (Python 3.12, `requirements.txt`). Tanpa folder itu, halaman awal dashboard meminta unggah `hasil_model.zip`.

## Isi proyek

- `config.py` — semua pengaturan model & tampilan
- `scripts/interactive.py` — representasi, DTW, PAM, DBA, FCM, MDS, metrik
- `scripts/distance.py` — FastDTW (numba)
- `scripts/model_store.py` — bangun grid, pilih model terbaik, ekspor/impor `hasil_model`
- `scripts/visualization.py`, `scripts/interactive_viz.py` — grafik Plotly (gaya factoextra)
- `scripts/dashboard_model.py`, `app.py` — dashboard Streamlit (membaca hasil, tidak melatih ulang)
