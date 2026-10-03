"""Формирование сабмишена: предсказания на тесте в формате Kaggle."""

import pandas as pd


def make_submission(model: object, X_test: pd.DataFrame, test_ids: pd.Series) -> pd.DataFrame:
    """Строит сабмишен ``id, FloodProbability`` по предсказаниям модели.

    Логика 1:1 из финальной ячейки ``Flood_final.ipynb``:
    ``comparison_df[['id', 'FloodProbability']]``.

    Args:
        model: Обученная модель с методом ``predict``.
        X_test: Тестовые признаки (те же, что и при обучении).
        test_ids: Серия с ``id`` тестовых объектов (из ``data/raw/test.csv``).

    Returns:
        DataFrame с колонками ``id`` и ``FloodProbability``.
    """
    df = pd.DataFrame({"id": test_ids})
    df["FloodProbability"] = model.predict(X_test)
    return df[["id", "FloodProbability"]]
