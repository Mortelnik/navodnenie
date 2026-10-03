"""Предобработка данных: приведение типов к компактным (downcast)."""

import pandas as pd


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит int-колонки к минимальному int-типу, float — к float32.

    Логика 1:1 перенесена из ячейки downcast в ``Flood_final.ipynb``:
    если тип содержит ``int`` — ``pd.to_numeric(..., downcast="integer")``,
    иначе — ``pd.to_numeric(..., downcast="float")``.

    Args:
        df: Исходный DataFrame.

    Returns:
        Копия DataFrame с уменьшенными типами.
    """
    result = df.copy()
    for col in result.columns:
        if "int" in str(result[col].dtype):
            result[col] = pd.to_numeric(result[col], downcast="integer")
        else:
            result[col] = pd.to_numeric(result[col], downcast="float")
    return result
