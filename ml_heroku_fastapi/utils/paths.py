from pyprojroot import here as BASE_DIR

DATA_DIR = BASE_DIR() / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR() / "model"
REPORTS_DIR = BASE_DIR() / "reports"
METRICS_DIR = BASE_DIR() / "metrics"