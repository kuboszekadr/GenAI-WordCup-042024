from pathlib import Path


# -----------------
# PATHS SECTION
# -----------------
ROOT_PATH = Path(__file__).parent.parent
DATASET_PATH = ROOT_PATH / "dataset" / "Azure Predictive Maintenance"

ERROR_DATA_PATH = DATASET_PATH / "errors.csv"
FAILURES_DATA_PATH = DATASET_PATH / "failures.csv"
MACHINES_DATA_PATH = DATASET_PATH / "machines.csv"
MAINTENANCE_DATA_PATH = DATASET_PATH / "maintenance.csv"
SENSORS_DATA_PATH = DATASET_PATH / "sensors.csv"

MODELS_PATH = ROOT_PATH / "store" / "models"
MULTICLASS_XGB_MODEL = MODELS_PATH / "xgboost_multiclass.json"

# -----------------
# DATABASE SECTION
# -----------------
TABLES = [
    SENSORS_DATA_PATH,
    ERROR_DATA_PATH,
    FAILURES_DATA_PATH,
    MACHINES_DATA_PATH,
    MAINTENANCE_DATA_PATH
]
SENSORS_FIELDS = ['volt', 'rotate', 'pressure', 'vibration']  # sensor columns to process

