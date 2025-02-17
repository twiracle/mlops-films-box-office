"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
import pandas as pd
import requests
import streamlit as st


def evaluate_input(unique_data_path: str, endpoint: str) -> None:
    """
    Получение результатов модели по введенным данным

    Parameters
    ----------
    unique_data_path
        путь к данным с уникальными значениями признаков
    endpoint
        эндпоинт с получением предсказаний

    Returns
    -------
    none
    """
    with open(unique_data_path, "r", encoding="utf-8") as file:
        unique_data = json.load(file)

    title = st.sidebar.text_input(label="Название:", max_chars=50)
    rating = st.sidebar.slider(label="Оценка:", max_value=10.0, min_value=0.0, step=0.1)
    production_year = st.sidebar.number_input(
        label="Год производства:",
        max_value=max(unique_data["production_year"]),
        min_value=min(unique_data["production_year"]),
    )
    release_date = st.sidebar.date_input(label="Дата релиза:", format="DD/MM/YYYY")
    duration = st.sidebar.number_input(label="Продолжительность (мин):", min_value=10)
    budget = st.sidebar.number_input(label="Бюджет ($):", min_value=0)
    unique_data["age_rating"].remove("undefined")
    age_rating = st.sidebar.selectbox(
        label="Возрастной рейтинг:", options=unique_data["age_rating"]
    )
    main_genre = st.sidebar.selectbox(label="Жанр:", options=unique_data["main_genre"])
    director = st.sidebar.text_input(label="Режиссер:", max_chars=50)
    cast = st.sidebar.text_area(label="Актерский состав:", max_chars=100)

    st.write(
        f"""
        #### Введенные данные:
        1) Название: {title}
        2) Оценка: {rating}
        3) Год производства: {production_year}
        4) Дата релиза: {release_date}
        5) Продолжительность (мин): {duration}
        6) Бюджет ($): {budget}
        7) Возрастной рейтинг: {age_rating}
        8) Жанр: {main_genre}
        9) Режиссер: {director}
        10) Актерский состав: {cast}
        """
    )

    dict_data = {
        "rating": rating,
        "production_year": production_year,
        "director": director,
        "age_rating": age_rating,
        "duration": duration,
        "budget": budget,
        "main_genre": main_genre,
        "month": release_date.month,
        "cast": cast.split(","),
    }

    if title != "" and director != "" and cast != "":
        if st.button("Predict"):
            result = requests.post(endpoint, timeout=5000, json=dict_data)
            output = result.json()
            st.write(f"#### {output}")


def evaluate_from_file(endpoint: str, files: dict) -> None:
    """
    Получение результатов модели по файлу с данными

    Parameters
    ----------
    data
        файл с данными
    endpoint
        эндпоинт с получением предсказаний
    files
        файлы для предсказания

    Returns
    -------
    none
    """
    predict_button = st.button("Predict")

    if predict_button:
        output = requests.post(endpoint, files=files, timeout=5000)
        prediction = output.json()["prediction"]
        st.write(pd.Series(prediction, name="Predictions"))
