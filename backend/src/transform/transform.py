"""
Программа: Предобработка данных
Версия: 1.1
"""

import json
import pandas as pd
import numpy as np
import warnings
import re
from typing import Optional
from ..data.get_data import save_data


warnings.filterwarnings("ignore")


def remove_symbols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Убирает ненужные символы и пробелы из ячеек столбцов, оставляя цифры

    Parameters
    ----------
    df
        дата фрейм с данными
    cols
        нужные столбцы

    Returns
    -------
    обработанный дата фрейм
    """
    for col in cols:
        df[col] = df[col].astype(str).str.replace(r"\D", "", regex=True)

    return df


def replace_values_with_none(df: pd.DataFrame, cols_values: dict) -> pd.DataFrame:
    """
    Замена значений в столбцах на None
    Parameters
    ----------
    df
        дата фрейм
    cols_values
        словарь со столбцами и данными для замены

    Returns
    -------
    обработанный дата фрейм
    """
    for col, value in cols_values.items():
        if value == "symbols":
            df[col] = df[col].astype(str).where(df[col].astype(str).str.isdigit(), None)
        else:
            df[col] = df[col].where(df[col] != value, None)

    return df


def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает таргет переменную (общие кассовые сборы) и сразу ее логарифмирует

    Parameters
    ---------
    df
        дата фрейм

    Returns
    ---------
    обработанный дата фрейм с целевой переменной
    """
    df["target"] = pd.to_numeric(df.box_office_usa).add(
        pd.to_numeric(df.box_office_world), fill_value=0
    )
    df["target_log"] = np.log(df.target + 1)
    return df


def fill_production_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные значения в столбце с годом производства годом релиза

    Parameters
    --------
    df
        дата фрейм

    Returns
    --------
    обработанный дата фрейм
    """
    df["production_year"] = df.production_year.fillna(
        df.release_date.str.extract(r"(\d{4})$")[0]
    )
    return df


def convert_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Переводит продолжительность фильма в нужную нам форму (только минуты)

    Parameters
    --------
    df
        дата фрейм

    Returns
    --------
    обработанный дата фрейм
    """
    hours = df.duration.str.extract(r"(\d+)\s*ч")[0].fillna(0).astype(int)
    minutes = df.duration.str.extract(r"(\d+)\s*мин")[0].fillna(0).astype(int)

    # Рассчитываем итоговую длительность в минутах
    df.duration = hours * 60 + minutes
    return df


def save_unique_values(
    df: pd.DataFrame, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение уникальных значений признаков

    Parameters
    -------------
    df
        дата фрейм с данными
    target_column
        название таргет переменной
    unique_values_path
        путь к файлу с уникальными значениями

    Returns
    --------------
    сохраняет уникальные значения признаков в файл
    """
    df_unique = df.drop(columns=target_column, axis=1)

    dict_unique = {key: df_unique[key].unique().tolist() for key in df_unique.columns}

    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


def add_main_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет новый признак - основной жанр

    Parameters
    ----------
    df
        дата фрейм с данными

    Returns
    -------
    обработанный дата фрейм
    """
    df["main_genre"] = df.genre.str.split(", ").str[0]

    return df


def transform_cast(df: pd.DataFrame, actors_fame_path: str) -> pd.DataFrame:
    """
    Добавляет новый признак - слава актеров

    Parameters
    ----------
    df
        дата фрейм с данными
    actors_fame_path
        путь к файлу со славой актеров

    Returns
    -------
    обработанный дата фрейм
    """
    df["cast"] = (
        df.cast.str.replace(r"[\'\[\]]", "", regex=True).str.split(", ").str[:3]
    )

    # разделение актеров на строки
    all_actors = df.cast.explode()
    # создадим словарь, где ключом будет имя актера, а значением - количество фильмов, в котором он снимался
    actor_counts = all_actors.value_counts().to_dict()
    # создадим новый признак - слава актеров
    df["actors_fame"] = df.cast.apply(
        lambda x: sum(actor_counts.get(actor, 0) for actor in x)
    )

    save_data_to_file(actor_counts, actors_fame_path)

    return df


def transform_director(
    df: pd.DataFrame, n_top: int, director_film_count_path: str, top_directors_path: str
) -> pd.DataFrame:
    """
    Преобразует признак director

    Parameters
    ----------
    df
        дата фрейм с данными
    n_top
        сколько первых режиссеров оставить
    director_film_count_path
        путь к файлу с количеством фильмов у режиссера
    top_directors_path
        путь к файлу с топом режиссеров

    Returns
    -------
    обработанный дата фрейм
    """
    df["director_film_count"] = df.groupby("director").title.transform("count")

    save_data_to_file(
        df.set_index("director")["director_film_count"].to_dict(),
        director_film_count_path,
    )

    top_directors = df.director.value_counts().nlargest(n_top).index
    # замена остальных значений на 'Other'
    df.director = df.director.where(df.director.isin(top_directors), other="Other")

    save_data_to_file(list(top_directors), top_directors_path)

    return df


def change_types(df: pd.DataFrame, cols_types: dict) -> pd.DataFrame:
    """
    Меняет типы данных у признаков

    Parameters
    ----------
    df
        дата фрейм с данными
    cols_types
        словарь с признаками и их типами данных

    Returns
    -------
    обработанный дата фрейм
    """
    for col, typ in cols_types.items():
        df[col] = df[col].astype(typ)

    return df


def fill_na_values(df: pd.DataFrame, cols_values: dict) -> pd.DataFrame:
    """
    Заполнение None значений
    Parameters
    ----------
    df
        дата фрейм с данными
    cols_values
        словарь с признаками и нужными значениями

    Returns
    -------
    обработанный дата фрейм
    """
    for col, value in cols_values.items():
        if value == "mean":
            df[col] = df[col].fillna(
                df.groupby("main_genre")[col].apply(lambda x: x.fillna(x.mean()))
            )
        else:
            df[col] = df[col].fillna(value)

    return df


def save_data_to_file(data: list | dict, data_path: str) -> None:
    """
    Сохранение данных в файл

    Parameters
    ----------
    data
        данные для сохранения
    data_path
        путь к данным

    Returns
    -------
    none
    """
    with open(data_path, "w", encoding="utf-8") as file:
        json.dump(data, file)


def get_data_from_file(data_path: str) -> list | dict:
    """
    Получает данные изз файла

    Parameters
    ----------
    data_path
        путь к файлу

    Returns
    -------
    данные в виде словаря либо списка
    """
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def preprocess_test(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Обработка тестовых данных
    Parameters
    ----------
    df
        дата фрейм, который нужно обработать

    Returns
    -------
    обработанный дата фрейм
    """
    top_directors = get_data_from_file(kwargs["top_directors_path"])
    film_count = get_data_from_file(kwargs["director_film_count_path"])
    actors_fame = get_data_from_file(kwargs["actors_fame_path"])

    df["director_film_count"] = df.director.map(film_count).fillna(1).astype(int)
    df["director"] = df.director.where(df.director.isin(top_directors), "Other")

    df["actors_fame"] = df.cast.apply(
        lambda x: sum(actors_fame.get(actor, 1) for actor in x)
    )

    print(df.info())

    if df.month.dtype == "object":
        print(df.month)
        df.month = df.month.map(kwargs["map_columns"]["month_names"]).fillna(df.month)
        print(df.month)
    else:
        df.month = df.month.map(kwargs["map_columns"]["month_numbers"])

    df = df.drop(["cast"], axis=1)
    df = change_types(df, kwargs["change_type_last"])

    return df


def preprocess_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Шаги по очистке и предобработке данных
    Parameters
    ----------
    df
        дата фрейм, который нужно обработать

    Returns
    -------
    обработанный дата фрейм
    """
    df = remove_symbols(df, kwargs["change_cols"]["money_cols"])
    df = df.dropna(subset=kwargs["drop_na_cols"]["drop_first"])
    df = replace_values_with_none(df, kwargs["change_values_to_none"])
    df = add_target_variable(df)
    df = df.dropna(subset=kwargs["drop_na_cols"]["drop_last"])

    df = fill_production_year(df)
    df = convert_duration(df)

    return df


def feature_engineering(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Шаги по созданию новых признаков
    Parameters
    ----------
    df
        дата фрейм, который нужно обработать

    Returns
    -------
    обработанный дата фрейм
    """
    df = add_main_genre(df)
    df["month"] = (
        df["release_date"]
        .str.split(" ")
        .str[1]
        .map(kwargs["map_columns"]["month_names"])
    )

    df = transform_director(
        df, 500, kwargs["director_film_count_path"], kwargs["top_directors_path"]
    )
    df = transform_cast(df, kwargs["actors_fame_path"])

    return df


def finalize_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Финальные шаги по обработке и сохранению данных
    Parameters
    ----------
    df
        дата фрейм, который нужно обработать

    Returns
    -------
    обработанный дата фрейм
    """
    df = change_types(df, kwargs["change_type_first"])
    df = fill_na_values(df, kwargs["fill_na_cols"])
    df = change_types(df, kwargs["change_type_last"])

    save_data(df, kwargs["clean_data_path"])
    df = df.drop(kwargs["drop_columns"], axis=1)

    return df


def generate_fillna_values(df: pd.DataFrame, path: str) -> None:
    """
    Генерация значений для заполнения пропусков

    Parameters
    ----------
    df
        исходный дата фрейм
    path : str
        путь для сохранения файла
    """
    fill_values = {}

    # определяем числовые и категориальные признаки
    numeric_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # заполнение для числовых признаков
    for col in numeric_cols:
        if col != "main_genre":
            fill_values[col] = df.groupby("main_genre")[col].mean().dropna().to_dict()

    # заполнение для категориальных признаков
    for col in cat_cols:
        fill_values[col] = df[col].mode().iloc[0]  # Мода для всего столбца

    save_data_to_file(fill_values, path)


def fill_missing_values_test(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Заполнение пропусков в тестовых данных значениями из файла

    Parameters
    ----------
    df : pd.DataFrame
        тестовые данные
    path : str
        путь к файлу со значениями для заполнения

    Returns
    -------
    обработанный датафрейм
    """
    fill_values = get_data_from_file(path)

    # Заполнение числовых признаков (по main_genre)
    for col, values in fill_values.items():
        if col in df.columns:
            if isinstance(values, dict):
                df[col] = df.apply(
                    lambda row: (
                        values.get(str(row["main_genre"]), row[col])
                        if pd.isna(row[col]) and row["main_genre"] in values
                        else row[col]
                    ),
                    axis=1,
                )

            else:
                if col in df.columns:
                    df[col] = df[col].fillna(values)

    return df


def pipeline_preprocessing(
    df: pd.DataFrame, flg_evaluation: bool = True, **kwargs
) -> pd.DataFrame:
    """
    Пайплайн для предварительной обработки данных
    Parameters
    ----------
    df
        дата фрейм, который нужно обработать
    flg_evaluation
        флаг, что мы прогнозируем результат

    Returns
    -------
    обработанный дата фрейм
    """
    if flg_evaluation:
        df = preprocess_test(df, **kwargs)
        if df.isna().sum().sum() > 0:
            df = fill_missing_values_test(df, kwargs["fillna_values_path"])

    else:
        # проведем первичную обработку данных
        df = preprocess_data(df, **kwargs)
        # создадим новые признаки
        df = feature_engineering(df, **kwargs)
        # проведем окончательную обработку данных
        df = finalize_data(df, **kwargs)
        # сохраним уникальные значения признаков
        save_unique_values(
            df=df,
            target_column=kwargs["target_column"],
            unique_values_path=kwargs["unique_values_path"],
        )
        generate_fillna_values(
            df.drop(["target_log"], axis=1),
            path=kwargs["fillna_values_path"],
        )

    return df
