"""
Программа: Бэкенд сервиса для отображения EDA, графиков, обучения модели
и прогнозирования кассовых сборов фильма по его данным
Версия: 1.0
"""

import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

import pandas as pd
import warnings
import optuna

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics_from_file


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/config.yaml"


class FilmData(BaseModel):
    """
    Данные фильма для получения прогнозов модели
    """

    rating: float
    production_year: int
    director: str
    age_rating: str
    duration: int
    budget: int
    main_genre: str
    month: str
    cast: list


@app.get("/")
def welcome() -> dict:
    """
    Эндпоинт для проверки

    Returns
    -------
    словарь с фразой для проверки
    """
    return {"message": "Hi! Nice to meet you."}


@app.post("/training")
def training():
    """
    Обучение модели и логирование метрик модели

    Returns
    -------
    словарь с метриками
    """
    pipeline_training(CONFIG_PATH)
    metrics = load_metrics_from_file(CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/prediction_file")
def prediction_file(file: UploadFile = File(...)):
    """
    Предсказание модели на данных из файла

    Parameters
    ----------
    file
        файл с данными
    Returns
    -------
    словарь с результатами предсказания
    """
    df = pd.read_csv(file.file)
    file.file.close()

    result = pipeline_evaluate(CONFIG_PATH, df)
    return {"prediction": result[:5]}


@app.post("/prediction_input")
def prediction_input(film: FilmData):
    """
    Предсказание модели на введенных данных

    Parameters
    ----------
    film
        введенные данные

    Returns
    -------
    словарь с результатами предсказаний
    """
    cols = [
        "rating",
        "production_year",
        "director",
        "age_rating",
        "duration",
        "budget",
        "main_genre",
        "month",
        "cast",
    ]
    features = [
        [
            film.rating,
            film.production_year,
            film.director,
            film.age_rating,
            film.duration,
            film.budget,
            film.main_genre,
            film.month,
            film.cast,
        ]
    ]

    film_data = pd.DataFrame(features, columns=cols)
    prediction = pipeline_evaluate(CONFIG_PATH, film_data)[0]

    result = f"Кассовые сборы фильма ~= {prediction}$"
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=80)
