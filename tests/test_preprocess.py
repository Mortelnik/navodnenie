"""Тесты downcast_dtypes: сокращение типов без изменения значений."""

import numpy as np
import pandas as pd

from navodnenie.data.preprocess import downcast_dtypes


def test_downcast_dtypes_reduces_int_and_float() -> None:
    """int64 → меньший int-тип, float64 → меньший float-тип, значения те же."""
    df = pd.DataFrame(
        {
            "a": pd.array([1, 2, 3], dtype="int64"),
            "b": pd.array([1.5, 2.5, 3.5], dtype="float64"),
        }
    )
    assert df["a"].dtype == np.int64
    assert df["b"].dtype == np.float64

    result = downcast_dtypes(df)

    # int64 должен стать int8
    assert result["a"].dtype == np.int8
    # float64 должен стать float32
    assert result["b"].dtype == np.float32
    # Значения не изменились
    pd.testing.assert_series_equal(result["a"].astype(np.int64), df["a"])
    np.testing.assert_allclose(result["b"].to_numpy(), df["b"].to_numpy())


def test_downcast_dtypes_does_not_modify_input() -> None:
    """Функция возвращает копию — исходный DataFrame не мутирует."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    _ = downcast_dtypes(df)
    assert df["a"].dtype == np.int64
