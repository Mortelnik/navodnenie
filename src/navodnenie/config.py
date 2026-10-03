"""Константы, пути и гиперпараметры проекта.

Все значения перенесены из `Flood_final.ipynb`, чтобы гарантировать
воспроизводимость исходных результатов (RANDOM_STATE = 2213).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Константы воспроизводимости
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 2213
TARGET_COLUMN: str = "FloodProbability"
TEST_SIZE: float = 0.2

# ---------------------------------------------------------------------------
# Пути (относительно корня репозитория)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

TRAIN_CSV: Path = RAW_DATA_DIR / "train.csv"
TEST_CSV: Path = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV: Path = RAW_DATA_DIR / "sample_submission.csv"

STACKING_MODEL_PATH: Path = MODELS_DIR / "stacking_final.joblib"
META_MODEL_PATH: Path = MODELS_DIR / "extreme_risk_meta.joblib"
FEATURE_ORDER_PATH: Path = MODELS_DIR / "feature_order.json"

# ---------------------------------------------------------------------------
# Имена исходных признаков риска (20 столбцов, все, кроме `id` и таргета)
# ---------------------------------------------------------------------------
RISK_COLUMNS: list[str] = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]

# Инженерные признаки, добавляемые `create_features`
SUM_RISK_COLUMN: str = "sum_risk"
STD_RISK_COLUMN: str = "std_risk"
EXTREME_RISK_COLUMN: str = "extreme_risk"

# Квантиль для бинаризации мета-признака extreme_risk
EXTREME_RISK_QUANTILE: float = 0.9

# ---------------------------------------------------------------------------
# Гиперпараметры моделей (значения из Flood_final.ipynb)
# ---------------------------------------------------------------------------
DT_MAX_DEPTH: int = 10
RF_N_ESTIMATORS: int = 100
RF_MAX_DEPTH: int = 12
CATBOOST_ITERATIONS: int = 1000
BOOST_LEARNING_RATE: float = 0.05
BOOST_N_ESTIMATORS: int = 1000
BOOST_MAX_DEPTH: int = 6

STACKING_CV: int = 3
RIDGE_ALPHA_GRID: list[float] = [1.0, 10.0, 100.0]
