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

PLOT_FONT = "Plus Jakarta Sans, Arial"
PLOT_PRIMARY = "#00806A"
PLOT_ACCENT = "#E06C2B"
PLOT_PALETTE = ["#00806A", "#E06C2B", "#3A6FD9", "#C8457F", "#7B5BD6"]
PLOT_NOISE = "#9AA39E"
PLOT_INK = "#13201B"
PLOT_MUTED = "#6B7770"
PLOT_GRID = "#EEEBE2"
PLOT_SEQUENTIAL = ["#F5F3EC", "#BFE3D6", "#55B196", "#00806A", "#00483A"]
PLOT_SEQUENTIAL_WARM = ["#FBEFE7", "#F6CFB6", "#EDA47C", "#E06C2B", "#A9481A"]
