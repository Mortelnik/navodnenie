"""Оценка моделей: метрики R2 / RMSE.

Реализации перенесены из ячеек ``evaluate_model`` и ``get_metrics``
в ``Flood_final.ipynb``. В отличие от оригинала, ``get_metrics`` принимает
``X_val``/``y_val`` явно, а не через глобальные переменные ноутбука.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    data_desc: str = "",
) -> float:
    """Обучает модель и считает R2 на валидационной выборке.

    Логика 1:1 из ``Flood_final.ipynb`` (кроме глобального ``results_list`` —
    метрики возвращаются и печатаются, хранение за вызывающим кодом).

    Args:
        model: Не обученная модель sklearn.
        X_train: Обучающие признаки.
        y_train: Обучающий таргет.
        X_val: Валидационные признаки.
        y_val: Валидационный таргет.
        model_name: Название модели для вывода.
        data_desc: Описание данных для вывода.

    Returns:
        Значение R2 на валидационной выборке.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    print(f"{model_name} ({data_desc}) -> R2: {r2:.5f}, RMSE: {rmse:.5f}")
    return r2


def get_metrics(
    searcher: BaseEstimator,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    name: str,
    duration: float,
) -> dict[str, float | str]:
    """Считает метрики для готового searcher (GridSearch и т.п.).

    Логика 1:1 из ``get_metrics`` в ``Flood_final.ipynb``, но ``X_val``/
    ``y_val`` передаются явно вместо глобальных переменных.

    Args:
        searcher: Обученный объект с методом ``predict``.
        X_val: Валидационные признаки.
        y_val: Валидационный таргет.
        name: Название метода для результата.
        duration: Время обучения в минутах.

    Returns:
        Словарь с метриками: ``Метод``, ``R2``, ``RMSE``, ``Время (мин)``.
    """
    y_pred = searcher.predict(X_val)
    return {
        "Метод": name,
        "R2": r2_score(y_val, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_val, y_pred))),
        "Время (мин)": duration,
    }


def time_inference(model: BaseEstimator, X: pd.DataFrame) -> float:
    """Измеряет время инференса модели на ``X`` в мс/объект.

    Логика из ячейки сравнения времени инференса в ``Flood_final.ipynb``.

    Args:
        model: Обученная модель.
        X: Данные для предсказания.

    Returns:
        Время на один объект в миллисекундах.
    """
    start = time.time()
    _ = model.predict(X)
    total = time.time() - start
    return (total / len(X)) * 1000
