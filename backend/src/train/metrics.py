"""
Программа: Получение метрик на данных
Версия: 1.1
"""

import yaml
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor


def r2_adjusted(y_true: pd.Series, y_pred: pd.Series, x_test: pd.DataFrame) -> float:
    """
    Вычисление коэффициента детерминации для множественной регрессии

    Parameters
    -----------
    y_test: np.ndarray
        тестовые значения y
    y_pred: np.ndarray
        предсказания модели
    x_test: np.ndarray
        тестовые значения X

    Returns
    ----------
    Значение метрики
    """
    n_objects = len(y_true)
    n_features = x_test.shape[1]
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n_objects - 1) / (n_objects - n_features - 1)


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Вычисление взвешенной абсолютной процентной ошибки

    Parameters
    -----------
    y_test: np.ndarray
        тестовые значения y
    y_pred: np.ndarray
        предсказания модели

    Returns
    ----------
    Значение метрики
    """
    return np.sum(np.abs(y_pred - y_true)) * 100 / np.sum(y_true)


def get_dict_metrics(
    y_test: pd.Series, y_pred: pd.Series, x_test: pd.DataFrame
) -> dict:
    """
    Создание таблицы с основными метриками для модели

    Parameters
    ----------
    y_test
        тестовые значения y
    y_pred
        предсказания модели
    x_test
        тестовые значения X

    Returns
    -------
    словарь с метриками
    """
    y_test_exp = np.exp(y_test) - 1
    y_pred_exp = np.exp(y_pred) - 1
    dict_metrics = {
        "MSE": mean_squared_error(y_test_exp, y_pred_exp),
        "MAE": mean_absolute_error(y_test_exp, y_pred_exp),
        "R2": r2_adjusted(y_test_exp, y_pred_exp, x_test),
        "WAPE_%": wape(y_test_exp, y_pred_exp),
    }

    return dict_metrics


def save_metrics_to_file(
    x_data: pd.DataFrame, y_data: pd.Series, model: LGBMRegressor, metric_path: str
) -> None:
    """
    Сохранение метрик в файл

    Parameters
    ----------
    x_data
        значения Х
    y_data
        значения у
    model
        модель
    metric_path
        путь к метрикам

    Returns
    -------
    ничего
    """
    result_metrics = get_dict_metrics(
        y_test=y_data, y_pred=model.predict(x_data), x_test=x_data
    )
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics_from_file(config_path: str) -> dict:
    """
    Загружает метрики из файла

    Parameters
    ----------
    config_path
        путь к конфигу

    Returns
    -------
    словарь с метриками
    """
    with open(config_path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics
