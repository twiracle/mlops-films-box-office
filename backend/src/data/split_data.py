"""
Программа: Разделение данных на train/validation/test
Версия: 1.0
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_val_test(
    df: pd.DataFrame, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделение данных на train/validation/test

    Parameters
    ----------
    df
        дата фрейм с данными

    Returns
    -------
    дата фреймы с тренировочными, валидационными и тестовыми данными
    """
    df_train_, df_test = train_test_split(
        df,
        test_size=kwargs["test_size"],
        shuffle=True,
        random_state=kwargs["random_state"],
    )

    df_train, df_val = train_test_split(
        df_train_,
        test_size=kwargs["val_size"],
        shuffle=True,
        random_state=kwargs["random_state"],
    )

    return df_train, df_val, df_test


def get_x_y_data(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Делит дата фрейм на X и y

    Parameters
    -----------
    df
        дата фрейм с данными
    target_col
        название таргет переменной

    Returns
    ----------
    разделенный дата фрейм на X и y
    """
    x = df.drop([target_col], axis=1)
    y = df[target_col]
    return x, y


def get_train_val_test_data(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Разбивает все дата фреймы на обьекты и признаки

    Parameters
    ----------
    df_train
        дата фрейм с тренировочными данными
    df_val
        дата фрейм с валидационными данными
    df_test
        дата фрейм с тестовыми данными
    target_col
        целевая переменная

    Returns
    -------
    наборы данных train/val/test
    """
    x_train, y_train = get_x_y_data(df_train, target_col)
    x_val, y_val = get_x_y_data(df_val, target_col)
    x_test, y_test = get_x_y_data(df_test, target_col)

    return x_train, y_train, x_val, y_val, x_test, y_test
