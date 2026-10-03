"""Обучение моделей: бейзлайны, stacking и подбор alpha.

Состав моделей и параметры перенесены из ячеек списка ``models``,
``StackingRegressor`` и GridSearchCV в ``Flood_final.ipynb``.
"""

from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from navodnenie.config import (
    BOOST_LEARNING_RATE,
    BOOST_MAX_DEPTH,
    BOOST_N_ESTIMATORS,
    CATBOOST_ITERATIONS,
    DT_MAX_DEPTH,
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RIDGE_ALPHA_GRID,
    STACKING_CV,
)


def get_baseline_models() -> list[tuple[str, Pipeline | object]]:
    """Возвращает список из 6 моделей-кандидатов (названия + объекты).

    Список 1:1 из ячейки ``models`` в ``Flood_final.ipynb``:
    Linear Regression, Decision Tree, Random Forest, CatBoost,
    XGBoost, LightGBM.
    """
    return [
        ("Linear Regression", make_pipeline(StandardScaler(), LinearRegression())),
        ("Decision Tree", DecisionTreeRegressor(max_depth=DT_MAX_DEPTH, random_state=RANDOM_STATE)),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "CatBoost",
            CatBoostRegressor(
                iterations=CATBOOST_ITERATIONS,
                learning_rate=BOOST_LEARNING_RATE,
                depth=BOOST_MAX_DEPTH,
                verbose=0,
                random_seed=RANDOM_STATE,
            ),
        ),
        (
            "XGBoost",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=BOOST_N_ESTIMATORS,
                            learning_rate=BOOST_LEARNING_RATE,
                            max_depth=BOOST_MAX_DEPTH,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ).set_output(transform="pandas"),
        ),
        (
            "LightGBM",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=BOOST_N_ESTIMATORS,
                            learning_rate=BOOST_LEARNING_RATE,
                            max_depth=BOOST_MAX_DEPTH,
                            random_state=RANDOM_STATE,
                            verbose=-1,
                        ),
                    ),
                ]
            ).set_output(transform="pandas"),
        ),
    ]


def build_stacking_model(
    lr: object,
    dt: object,
    rf: object,
    cat: object,
) -> Pipeline:
    """Собирает пайплайн StackingRegressor (LR, DT, RF, CatBoost → Ridge).

    Структура 1:1 из ячейки stacking в ``Flood_final.ipynb``:
    ``StandardScaler`` → ``StackingRegressor`` (cv=3, final=Ridge).
    """
    estimators = [
        ("lr", lr),
        ("dt", dt),
        ("rf_base", rf),
        ("cat", cat),
    ]
    stacking_logic = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=STACKING_CV,
    )
    return Pipeline([("scaler", StandardScaler()), ("stacking", stacking_logic)])


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    """Обучает финальную модель: stacking + GridSearchCV по ``alpha``.

    Логика 1:1 из ячейки GridSearchCV в ``Flood_final.ipynb``:
    сетка ``stacking__final_estimator__alpha`` = [1.0, 10.0, 100.0], cv=2, r2.

    Args:
        X_train: Обучающие признаки (с инженерными признаками).
        y_train: Обучающий таргет.

    Returns:
        ``best_estimator_`` из GridSearchCV (готовая к ``predict`` модель).
    """
    models = get_baseline_models()
    lr, dt, rf, cat = models[0][1], models[1][1], models[2][1], models[3][1]
    final_main_pipeline = build_stacking_model(lr, dt, rf, cat)

    param_grid = {"stacking__final_estimator__alpha": RIDGE_ALPHA_GRID}
    grid_final = GridSearchCV(final_main_pipeline, param_grid, cv=2, scoring="r2", n_jobs=-1)
    grid_final.fit(X_train, y_train)
    return grid_final.best_estimator_


def save_model(model: object, path: Path | str) -> Path:
    """Сохраняет обученную модель через joblib.

    Args:
        model: Обученная модель sklearn.
        path: Путь к файлу (например, ``models/stacking_final.joblib``).

    Returns:
        Путь, куда модель сохранена.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path | str) -> object:
    """Загружает модель, сохранённую через :func:`save_model`.

    Args:
        path: Путь к файлу модели.

    Returns:
        Обученная модель.
    """
    return joblib.load(path)


def build_poly_baseline() -> Pipeline:
    """Бейзлайн: StandardScaler + PolynomialFeatures(degree=2) + LinearRegression.

    Логика из ячейки ``pipeline_poly`` в ``Flood_final.ipynb``.
    """
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    )


def build_elasticnet_pipeline() -> Pipeline:
    """Бейзлайн: StandardScaler + ElasticNet (параметры из ноутбука)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elasticnet",
                ElasticNet(
                    random_state=RANDOM_STATE,
                    max_iter=500,
                    tol=0.01,
                    selection="random",
                    l1_ratio=0.01,
                    alpha=0.001,
                ),
            ),
        ]
    )
