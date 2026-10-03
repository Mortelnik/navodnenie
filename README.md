# Flood Prediction / Прогнозирование наводнений

Проект является проектной практикой студентов УрФУ.

**Описание проекта**

Проект посвящён разработке модели предсказания вероятности наводнения
на основе данных Kaggle (Playground Series S4E5). Основной фокус —
оптимизация: достижение наивысшего R2 score при минимальных
вычислительных затратах.

В ходе исследования было выявлено, что 3 синтетических признака
(`sum_risk`, `std_risk` и мета-фича `extreme_risk`, обученная на
логистической регрессии через `predict_proba`) объясняют дисперсию
данных лучше, чем 20 исходных факторов. Итого обучены порядка 9 моделей;
финальная — `StackingRegressor` (LR, Decision Tree, Random Forest,
CatBoost → Ridge).

## Структура репозитория

```text
src/navodnenie/          # ML-пакет (production-структура)
    config.py            # константы, пути, гиперпараметры (RANDOM_STATE = 2213)
    data/
        load.py          # загрузка локальных train/test/sample_submission
        preprocess.py    # downcast_dtypes
    features/
        build_features.py  # create_features, add_extreme_risk (OOF)
    models/
        train.py         # бейзлайны, stacking, подбор alpha, save/load модели
        evaluate.py      # evaluate_model, get_metrics
        predict.py       # make_submission
scripts/
    train.py             # end-to-end обучение (--sample N для быстрых проверок)
    predict.py           # предсказание на тесте → submission.csv
tests/                   # pytest: downcast, create_features (синтетические данные)
configs/                 # конфиги (пока пусто — см. src/navodnenie/config.py)
models/                  # сохранённые модели (игнорируются Git)
data/raw/                # сырые данные (отслеживаются — по ним воспроизводится пайплайн)
data/processed/          # подготовленные данные (не отслеживаются)
reports/                 # отчёты и графики
Flood_final.ipynb        # исходный ноутбук исследования (исторический)
```

## Установка

Проект использует [Poetry](https://python-poetry.org/).
Виртуальное окружение создаётся в проекте (`.venv/`) и **не коммитится** —
воспроизводимость окружения обеспечивает коммит
`pyproject.toml` + `poetry.lock` + `poetry.toml` (см. `poetry config virtualenvs.in-project true`).

```powershell
git clone https://github.com/Mortelnik/navodnenie
cd navodnenie
poetry install
poetry run pre-commit install
```

## Запуск

```powershell
# Обучение на полной выборке
poetry run python scripts/train.py

# Быстрая проверка на первых 1000 строк
poetry run python scripts/train.py --sample 1000

# Предсказание на тесте → submission.csv
poetry run python scripts/predict.py
```

Исследовательский вариант пайплайна — в `Flood_final.ipynb`
(`poetry run jupyter notebook Flood_final.ipynb`).

## Проверки качества

```powershell
poetry run ruff check .
poetry run pre-commit run --all-files
poetry run pytest
```

## История изменений

- Рефакторинг под production: src-layout, модули, конфиг, тесты,
  Poetry, pre-commit + Ruff.
