"""Тесты create_features: инженерные признаки sum_risk и std_risk."""

import numpy as np
import pandas as pd

from navodnenie.features.build_features import create_features


def _make_sample_df() -> pd.DataFrame:
    """DataFrame 5x21: id, FloodProbability и 20 риск-столбцов.

    Риск-столбцы заполнены последовательными числами 1..10 по строкам
    (5 строк x 20 колонок), чтобы сумму и std можно было посчитать вручную.
    """
    risk_cols = [f"risk_{i}" for i in range(20)]
    data = {
        "id": [10, 11, 12, 13, 14],
        "FloodProbability": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    # Каждая строка i: значения (i*20 + 1) .. (i*20 + 20)
    for i, col in enumerate(risk_cols):
        data[col] = [row * 20 + (i + 1) for row in range(5)]
    return pd.DataFrame(data)


def test_create_features_columns_and_shape() -> None:
    """Создаются ровно sum_risk и std_risk; id/таргет не попадают в результат."""
    df = _make_sample_df()
    result = create_features(df)

    assert list(result.columns) == ["sum_risk", "std_risk"]
    assert result.shape == (5, 2)
    # Индексы сохраняются
    pd.testing.assert_index_equal(result.index, df.index)


def test_create_features_values_hand_computed() -> None:
    """Значения проверяются на вручную посчитанном примере.

    Строка 0: значения 1..20 → sum = 210, std (sample, ddof=1) = 5.916079783099616.
    Строка 1: значения 21..40 → sum = 610, std = 5.916079783099616
    (std сдвига по константе не меняется).
    """
    df = _make_sample_df()
    result = create_features(df)

    expected_sum_row0 = sum(range(1, 21))  # = 210
    assert result.loc[0, "sum_risk"] == expected_sum_row0
    expected_sum_row1 = sum(range(21, 41))  # = 610
    assert result.loc[1, "sum_risk"] == expected_sum_row1

    # pandas .std по умолчанию считает sample-std (ddof=1)
    manual_std = np.std(np.arange(1, 21), ddof=1)
    assert np.isclose(result.loc[0, "std_risk"], manual_std)
    assert np.isclose(result.loc[1, "std_risk"], manual_std)

    # Сравнение с прямым pandas-вычислением по всем строкам
    risk_cols = [c for c in df.columns if c not in ("id", "FloodProbability")]
    pd.testing.assert_series_equal(result["sum_risk"], df[risk_cols].sum(axis=1), check_names=False)
    pd.testing.assert_series_equal(result["std_risk"], df[risk_cols].std(axis=1), check_names=False)
