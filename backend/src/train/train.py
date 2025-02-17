"""
Программа: Тренировка данных
Версия: 1.0
"""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from sklearn.model_selection import KFold
import optuna
from ..data.split_data import get_train_val_test_data
from ..train.metrics import save_metrics_to_file

import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_lgb(
    trial,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int,
    cat_features: list,
):
    """
    Функция для подбора параметров с помощью optuna

    Parameters
    ------------
    trial
        текущая попытка подбора
    x_train
        тренировочные значения X
    y_train
        тренировочные значения y
    x_val
        валидационные значения X
    y_val
        валидационные значения y
    random_state
    cat_features
        список с категориальными переменными в наборе

    Returns
    -----------
    MAE на текущем trial
    """
    lgb_params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [857]),
        #         trial.suggest_int("n_estimators", 300, 1000),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.045309611206922125]
        ),
        #                 trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "max_bin": trial.suggest_int("max_bin", 50, 500),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 150),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 20),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 20),
        "min_split_gain": trial.suggest_int("min_split_gain", 0, 20),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "objective": trial.suggest_categorical("objective", ["mae"]),
        "verbosity": trial.suggest_categorical("verbosity", [-1]),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
    }

    model = LGBMRegressor(**lgb_params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="mae",
        categorical_feature=cat_features,
        callbacks=[early_stopping(stopping_rounds=100)],
    )

    y_pred = model.predict(x_val)

    return mean_absolute_error(np.exp(y_val), np.exp(y_pred))


def find_best_params(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, **kwargs
) -> optuna.Study:
    """
    Поиск оптимальных параметров модели

    Parameters
    ----------
    df_train
        тренировочные данные
    df_val
        валидационные данные
    df_test
        тестовые данные

    Returns
    -------
    Study с параметрами модели
    """
    x_train, y_train, x_val, y_val, x_test, y_test = get_train_val_test_data(
        df_train, df_val, df_test, kwargs["target_column"]
    )

    study_lgbm = optuna.create_study(direction="minimize", study_name="lgbm")

    func = lambda trial: objective_lgb(
        trial,
        x_train,
        y_train,
        x_val,
        y_val,
        random_state=kwargs["random_state"],
        cat_features=kwargs["category_cols"],
    )

    study_lgbm.optimize(func, n_trials=kwargs["n_trials"], show_progress_bar=True)

    return study_lgbm


def train_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    study: optuna.Study,
    **kwargs
) -> LGBMRegressor:
    """
    Обучение модели

    Parameters
    ----------
    df_train
        тренировочные данные
    df_val
        валидационные данные
    df_test
        тестовые данные
    study
        study optuna

    Returns
    -------
    обученная модель
    """
    x_train, y_train, x_val, y_val, x_test, y_test = get_train_val_test_data(
        df_train, df_val, df_test, kwargs["target_column"]
    )

    model = LGBMRegressor(**study.best_params)
    model.fit(
        x_train,
        y_train,
        eval_metric="mae",
        categorical_feature=kwargs["category_cols"],
        eval_set=[(x_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=100)],
    )

    save_metrics_to_file(
        x_data=x_test, y_data=y_test, model=model, metric_path=kwargs["metrics_path"]
    )

    return model
