"""Точка входа: end-to-end обучение модели прогнозирования наводнения.

Пайплайн повторяет логику ``Flood_final.ipynb``:
загрузка локальных данных → downcast → сплит 80/20 → инженерные признаки
→ мета-признак extreme_risk (OOF) → обучение stacking-модели с подбором
alpha → оценка на валидации → сохранение модели в ``models/``.

Запуск:
    poetry run python scripts/train.py                # полная выборка
    poetry run python scripts/train.py --sample 1000  # быстрая проверка
"""

import argparse
import sys
from pathlib import Path

# Позволяет запускать скрипт без установки пакета: добавляем src/ в sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.model_selection import train_test_split  # noqa: E402

from navodnenie.config import (  # noqa: E402
    META_MODEL_PATH,
    RANDOM_STATE,
    STACKING_MODEL_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_CSV,
)
from navodnenie.data.load import load_train_data  # noqa: E402
from navodnenie.data.preprocess import downcast_dtypes  # noqa: E402
from navodnenie.features.build_features import (  # noqa: E402
    add_extreme_risk,
    create_features,
)
from navodnenie.models.evaluate import evaluate_model  # noqa: E402
from navodnenie.models.train import save_model, train_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Обучение модели наводнения")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Использовать только первые N строк (быстрая проверка)",
    )
    return parser.parse_args()


def run(sample: int | None = None) -> None:
    """Запускает полный пайплайн обучения."""
    df_train = load_train_data(TRAIN_CSV)
    if sample is not None:
        df_train = df_train.head(sample)
        print(f"Режим --sample {sample}: обучающая выборка {len(df_train)} строк")

    df_train = downcast_dtypes(df_train)

    X_full = df_train.drop(TARGET_COLUMN, axis=1)
    y_full = df_train[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
    )
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер валидационной выборки: {X_val.shape}")

    # Инженерные признаки по 20 столбцам риска
    X_train_opt = create_features(X_train)
    X_val_opt = create_features(X_val)

    # Мета-признак extreme_risk (OOF)
    X_train_opt, X_val_opt, _, meta_pipe = add_extreme_risk(
        X_train_opt, y_train, X_val_opt, X_val_opt
    )

    # Обучение stacking-модели с подбором alpha финальной регрессии
    model = train_model(X_train_opt, y_train)
    evaluate_model(model, X_train_opt, y_train, X_val_opt, y_val, "Stacking", "3 features")

    save_model(model, STACKING_MODEL_PATH)
    print(f"Модель сохранена: {STACKING_MODEL_PATH}")
    save_model(meta_pipe, META_MODEL_PATH)
    print(f"Мета-модель (extreme_risk) сохранена: {META_MODEL_PATH}")


def main() -> None:
    """Точка входа CLI."""
    args = parse_args()
    run(sample=args.sample)


if __name__ == "__main__":
    main()
