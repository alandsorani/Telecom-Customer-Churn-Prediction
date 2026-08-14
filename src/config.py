"""Central configuration: paths, filenames, and reproducibility settings.

Every notebook and script imports from here so that file locations and the
random seed are defined in exactly one place.
"""

from pathlib import Path

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SQL_DIR = PROJECT_ROOT / "sql"

RAW_DATA_FILE = DATA_RAW / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_DATA_FILE = DATA_PROCESSED / "telco_churn_clean.csv"
SQLITE_DB_FILE = DATA_PROCESSED / "telco_churn.db"
FINAL_MODEL_FILE = MODELS_DIR / "final_churn_pipeline.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Decision threshold for the final model, chosen in notebook 06 by maximizing
# F1 on out-of-fold TRAINING predictions (never on the test set).
DECISION_THRESHOLD = 0.338

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
