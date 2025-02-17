"""
Программа: Импорт данных из файла
Версия: 1.0
"""

import pandas as pd
from typing import Text


def get_data(path: Text) -> pd.DataFrame:
    """
    Импорт данных по заданному пути
    Parameters
    ----------
    path
        путь к данным

    Returns
    -------
    дата фрейм с данными
    """
    return pd.read_csv(path)


def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Сохраняет дата фрейм в файл
    Parameters
    ----------
    df
        дата фрейм для сохранения
    path
        путь к файлу, в который нужно сохранить

    Returns
    -------
    none
    """
    df.to_csv(path, index=False)
