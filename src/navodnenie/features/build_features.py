"""Формирование признаков: `create_features` и мета-признак `extreme_risk`.

Реализации перенесены без изменений из ячеек ``create_features`` и
OOF-блока ``Flood_final.ipynb``.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from navodnenie.config import (
    EXTREME_RISK_COLUMN,
    EXTREME_RISK_QUANTILE,
    RISK_COLUMNS,
    STD_RISK_COLUMN,
    SUM_RISK_COLUMN,
)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт агрегированные признаки ``sum_risk`` и ``std_risk``.

    Логика 1:1 из ``Flood_final.ipynb``: по всем колонкам DataFrame,
    кроме ``id`` и ``FloodProbability`` (в датасете это 20 столбцов
    риска), считаются сумма и стандартное отклонение по строке.

    Args:
        df: DataFrame с исходными столбцами риска
            (допустимы ``id`` и ``FloodProbability`` — они отбрасываются).

    Returns:
        DataFrame с колонками ``sum_risk`` и ``std_risk``.
    """
    cols = [c for c in df.columns if c not in ["id", "FloodProbability"]]
    new_df = pd.DataFrame(index=df.index)
    new_df[SUM_RISK_COLUMN] = df[cols].sum(axis=1)
    new_df[STD_RISK_COLUMN] = df[cols].std(axis=1)
    return new_df


def _risk_columns(df: pd.DataFrame) -> list[str]:
    """Возвращает колонки риска из ``df`` (все, кроме id и таргета)."""
    return [c for c in df.columns if c not in ["id", "FloodProbability"]]


def add_extreme_risk(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """Добавляет мета-признак ``extreme_risk`` с OOF-предсказанием.

    Логика 1:1 из ``Flood_final.ipynb``:
    1. бинаризация — выборки выше 90-го квантиля и таргета, и суммы риска;
    2. ``cross_val_predict`` (``StandardScaler`` + ``LogisticRegression``,
       ``method="predict_proba"``) даёт train-значения без утечки;
    3. финальная модель дообучается на всём train и применяет
       ``predict_proba`` к валидационной и тестовой выборкам.

    Args:
        X_train: Признаки train (результат ``create_features``), без
            ``extreme_risk``.
        y_train: Таргет train (``FloodProbability``).
        X_val: Признаки validation (результат ``create_features``).
        X_test: Признаки test (результат ``create_features``).

    Returns:
        Кортеж ``(X_train, X_val, X_test, meta_pipe)`` — тот же DataFrame с
        добавленной колонкой ``extreme_risk`` и обученная мета-модель
        (нужна для предсказания на новых данных в ``scripts/predict.py``).
    """
    features_to_use = [col for col in X_train.columns if col != EXTREME_RISK_COLUMN]

    q_target = y_train.quantile(EXTREME_RISK_QUANTILE)
    q_risk = X_train[SUM_RISK_COLUMN].quantile(EXTREME_RISK_QUANTILE)
    y_binary = ((y_train > q_target) & (X_train[SUM_RISK_COLUMN] > q_risk)).astype(int)

    meta_pipe = make_pipeline(StandardScaler(), LogisticRegression())
    X_train = X_train.copy()
    X_train[EXTREME_RISK_COLUMN] = cross_val_predict(
        meta_pipe,
        X_train[features_to_use],
        y_binary,
        cv=5,
        method="predict_proba",
    )[:, 1]

    meta_pipe.fit(X_train[features_to_use], y_binary)

    X_val = X_val.copy()
    X_val[EXTREME_RISK_COLUMN] = meta_pipe.predict_proba(X_val[features_to_use])[:, 1]
    X_test = X_test.copy()
    X_test[EXTREME_RISK_COLUMN] = meta_pipe.predict_proba(X_test[features_to_use])[:, 1]

    return X_train, X_val, X_test, meta_pipe


def get_model_feature_names(n_features: int) -> list[str]:
    """Возвращает ожидаемый порядок признаков финальной модели.

    Финальная модель обучается на ``sum_risk``, ``std_risk`` и
    ``extreme_risk`` — порядок фиксирован (см. ``create_features`` и
    ``add_extreme_risk``).

    Args:
        n_features: Ожидаемое число признаков (проверка целостности).

    Returns:
        Список имён признаков в порядке, в котором модель их ждёт.
    """
    names = [SUM_RISK_COLUMN, STD_RISK_COLUMN, EXTREME_RISK_COLUMN]
    if n_features is not None and n_features != len(names):
        # TODO: если модель обучена на ином наборе признаков, порядок
        #  следует сохранять вместе с моделью и читать в predict.
        raise ValueError(f"Ожидалось {len(names)} признаков, получено {n_features}")
    return names


def risk_columns() -> list[str]:
    """Возвращает список 20 исходных столбцов риска (из ``config``)."""
    return list(RISK_COLUMNS)
