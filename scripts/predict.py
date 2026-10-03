"""Точка входа: предсказание на тесте и запись submission.csv.

Загружает модель из ``models/``, готовит тестовые данные
(downcast + инженерные признаки + extreme_risk) и записывает
``submission.csv`` в корне репозитория.

Запуск:
    poetry run python scripts/predict.py
"""

import sys
from pathlib import Path

# Позволяет запускать скрипт без установки пакета: добавляем src/ в sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from navodnenie.config import (  # noqa: E402
    META_MODEL_PATH,
    STACKING_MODEL_PATH,
    TEST_CSV,
)
from navodnenie.data.load import load_test_data  # noqa: E402
from navodnenie.data.preprocess import downcast_dtypes  # noqa: E402
from navodnenie.features.build_features import create_features  # noqa: E402
from navodnenie.models.predict import make_submission  # noqa: E402
from navodnenie.models.train import load_model  # noqa: E402

SUBMISSION_PATH = Path(__file__).resolve().parents[1] / "submission.csv"


def run() -> None:
    """Загружает модель, готовит тест и пишет сабмишен."""
    df_test = downcast_dtypes(load_test_data(TEST_CSV))
    test_ids = df_test["id"]
    X_test = df_test.drop("id", axis=1)

    # Инженерные признаки по 20 столбцам риска
    X_test_opt = create_features(X_test)

    # Мета-признак: дообученная мета-модель (см. add_extreme_risk)
    meta_pipe = load_model(META_MODEL_PATH)
    X_test_opt = X_test_opt.copy()
    X_test_opt["extreme_risk"] = meta_pipe.predict_proba(X_test_opt)[:, 1]

    model = load_model(STACKING_MODEL_PATH)
    submission = make_submission(model, X_test_opt, test_ids)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Сабмишен сохранён: {SUBMISSION_PATH} ({len(submission)} строк)")


def main() -> None:
    """Точка входа CLI."""
    run()


if __name__ == "__main__":
    main()
