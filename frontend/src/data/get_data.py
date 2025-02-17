"""
Программа: Импорт данных из файла и чтение
Версия: 1.0
"""

import pandas as pd
from typing import Text
import streamlit as st
import io


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


def load_data(data: str, data_type: str) -> tuple[pd.DataFrame, dict]:
    """
    Получение данных и подготовка их для обработки в streamlit

    Parameters
    ----------
    data
        данные
    data_type
        тип данных (для тренировки либо для тестирования)

    Returns
    -------
    обработанный датасет
    """
    df = pd.read_csv(data)

    st.write("Данные были загружены")
    st.write(df.head())

    df_bytes = io.BytesIO()
    df.to_csv(df_bytes, index=False)
    df_bytes.seek(0)

    files = {"file": (f"{data_type}_dataset.csv", df_bytes, "multipart/form-data")}

    return df, files
