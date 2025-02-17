"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history


def train_model(config: dict, endpoint: str) -> None:
    """
    Тренировка модели и вывод метрик и результатов

    Parameters
    ----------
    config
        конфиг файл
    endpoint
        эндпоинт с тренировкой модели

    Returns
    -------
    none
    """
    try:
        with open(config["train"]["metrics_path"], "r") as file:
            old_metrics = json.load(file)
    except FileNotFoundError:
        old_metrics = {"MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0, "WAPE_%": 0}

    with st.spinner("Обучаем модель..."):
        output = requests.post(endpoint, timeout=5000)
    st.success("Модель успешно обучена!")

    new_metrics = output.json()["metrics"]

    mse, mae, r2, wape = st.columns(4)
    mse.metric(
        "MSE",
        f"{new_metrics['MSE']:.2e}",
        f"{new_metrics['MSE']-old_metrics['MSE']:.2e}",
    )
    mae.metric(
        "MAE",
        f"{new_metrics['MAE']:.2e}",
        f"{new_metrics['MAE']-old_metrics['MAE']:.2e}",
    )
    r2.metric(
        "R2", round(new_metrics["R2"], 4), f"{new_metrics['R2']-old_metrics['R2']:.4f}"
    )
    wape.metric(
        "WAPE_%",
        round(new_metrics["WAPE_%"], 2),
        f"{new_metrics['WAPE_%']-old_metrics['WAPE_%']:.2f}",
    )

    optuna_study = joblib.load(config["train"]["study_path"])
    optim_history = plot_optimization_history(optuna_study)
    st.plotly_chart(optim_history, use_container_width=True)
