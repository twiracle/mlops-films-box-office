"""
Программа: Получение предсказания на основе обученной модели
Версия: 1.0
"""

import yaml
import joblib
import pandas as pd
import numpy as np
from ..transform.transform import pipeline_preprocessing


def pipeline_evaluate(config_path: str, df: pd.DataFrame) -> list:
    """
    Обработка входных данных и получение предсказаний
    Parameters
    ----------
    config_path
        путь к конфигу
    df
        дата фрейм с данными

    Returns
    -------
    предсказания модели
    """
    with open(config_path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config_preproc = config["preprocessing"]
    config_train = config["train"]

    df = pipeline_preprocessing(df, **config_preproc)

    model = joblib.load(config_train["model_path"])
    prediction = np.round(np.exp(model.predict(df)), 2).tolist()

    return prediction
