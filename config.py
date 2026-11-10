"""Konfigurasi terpusat: pembangunan model (Colab) dan dashboard (Streamlit)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "hasil_model"          # hasil dari notebook Colab (diekstrak dari hasil_model.zip)
RESULTS_FILE = RESULTS_DIR / "model_results.json"
CACHE_DIR = PROJECT_ROOT / "cache"                   # matriks jarak FastDTW (tidak perlu di-commit)

ID_COLUMN = "Product_Code"
WEEK_COLUMNS = [f"W{i}" for i in range(52)]

# ---------------- Pembangunan model (notebook Colab) ----------------
REPRESENTATIONS = ["original", "minmax", "zscore"]  # Data asli · Min-Max per produk · Scale (z-score per produk)
DISTANCE = "fastdtw"
FASTDTW_RADIUS = 2
K_GRID = [2, 3, 4, 5, 6]
LINKAGES = ["average", "complete"]
DBSCAN_MIN_SAMPLES = 7
DBSCAN_EPS_PERCENTILES = [30, 50, 70, 85, 95]        # persentil jarak tetangga ke-min_samples
FUZZINESS = 2.0
RANDOM_STATE = 42
KMEANS_N_INIT = 1

# Aturan memilih "model terbaik" per representasi (di atas peringkat Silhouette↑ · DBI↓ · CHI↑)
MIN_BALANCE = 0.50     # tolak partisi "satu cluster raksasa"
MAX_NOISE_SHARE = 0.30  # DBSCAN: maksimal 30% produk noise
MIN_CLUSTERS = 2

# ---------------- Dashboard ----------------
DEFAULT_REPRESENTATION = "minmax"

# Tema "Dark Analytics" (navy · electric cyan · golden yellow).
# Warna cluster = langkah lebih dalam dari cyan/emas/emerald agar tervalidasi CVD & kontras di latar #0D2540.
PLOT_FONT = "Inter, Plus Jakarta Sans, Arial"
PLOT_PRIMARY = "#00D5F2"      # electric cyan — seri utama
PLOT_ACCENT = "#FFD166"       # golden yellow — sorotan / pembanding
PLOT_EMERALD = "#00C9A7"
PLOT_PALETTE = ["#0A9CC9", "#B88A12", "#1FA884", "#8E72EA", "#E35D8A"]   # C0..C4 (urutan tetap)
PLOT_NOISE = "#718BA3"
PLOT_INK = "#EAF2FA"
PLOT_MUTED = "#A9BDCF"
PLOT_GRID = "rgba(22,73,106,0.55)"
PLOT_SURFACE = "#0D2540"
PLOT_PAPER = "#0D2540"
PLOT_NEUTRAL = "#4A6B86"
PLOT_ALERT = "#FF647C"
PLOT_HOVER_BG = "#0B2945"
PLOT_SEQUENTIAL = ["#0D2540", "#0E3A5C", "#0B5E86", "#0A80AE", "#0AA6D6"]
PLOT_SEQUENTIAL_WARM = ["#0D2540", "#3A3524", "#6E5A24", "#A68325", "#E0B341"]
