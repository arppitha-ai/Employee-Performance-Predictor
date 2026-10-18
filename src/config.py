from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "employee_data.csv"
MODEL_PATH = ROOT / "models" / "employee_perf_model.pkl"
OUT_DIR = ROOT / "outputs"
IMG_DIR = ROOT / "images"

TARGET = "perf_band_next"
ORDER = ["Low", "Medium", "High"]
SEED = 13

for d in (DATA_PATH.parent, MODEL_PATH.parent, OUT_DIR, IMG_DIR):
    d.mkdir(parents=True, exist_ok=True)
