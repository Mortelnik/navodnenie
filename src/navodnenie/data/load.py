"""Загрузка сырых данных проекта с локальных путей.

В отличие от исходного `Flood_final.ipynb` (где данные читались по
hardcoded GitHub raw-URL), здесь используются локальные файлы из
`data/raw/` — они закоммичены в репозиторий и гарантируют
воспроизводимость.
"""

from pathlib import Path

import pandas as pd


def load_train_data(path: Path | str) -> pd.DataFrame:
    """Читает обучающую выборку (train.csv).

    Ожидаемые колонки: ``id``, 20 столбцов риска и ``FloodProbability``.

    Args:
        path: Путь к train.csv (по умолчанию — локальный файл репозитория).

    Returns:
        DataFrame со всей обучающей выборкой, включая ``id`` и таргет.
    """
    return pd.read_csv(path)


def load_test_data(path: Path | str) -> pd.DataFrame:
    """Читает тестовую выборку (test.csv).

    Ожидаемые колонки: ``id`` и 20 столбцов риска (таргета нет).

    Args:
        path: Путь к test.csv (по умолчанию — локальный файл репозитория).

    Returns:
        DataFrame с тестовой выборкой, включая ``id``.
    """
    return pd.read_csv(path)


def load_sample_submission(path: Path | str) -> pd.DataFrame:
    """Читает образцовый сабмишен (sample_submission.csv).

    Ожидаемые колонки: ``id`` и ``FloodProbability``.

    Args:
        path: Путь к sample_submission.csv.

    Returns:
        DataFrame формата сабмишена Kaggle.
    """
    return pd.read_csv(path)
