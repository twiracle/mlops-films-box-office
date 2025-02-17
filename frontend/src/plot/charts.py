"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def bar_plot(
    df: pd.DataFrame, x_col: str, y_col: str, title: str
) -> matplotlib.figure.Figure:
    """
    Построение bar_plot

    Parameters
    ----------
    df
        дата фрейм с данными
    x_col
        признак по х
    y_col
        признак по у
    title
        название графика

    Returns
    -------
    график
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.barplot(data=df, y=y_col, x=x_col, palette="rocket", hue=x_col, legend=False)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_col, fontsize=14)
    ax.set_ylabel(y_col, fontsize=14)
    plt.xticks(rotation=90)

    return fig


def correlation_plot(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Построение графика с корреляцией

    Parameters
    ----------
    df
        дата фрейм с данными

    Returns
    -------
    график
    """
    fig, ax = plt.subplots(ncols=2, figsize=(15, 6))

    sns.heatmap(df.corr(numeric_only=True), annot=True, linewidths=0.7, ax=ax[0])

    sns.heatmap(
        df.corr(method="spearman", numeric_only=True),
        annot=True,
        linewidths=0.7,
        ax=ax[1],
    )

    ax[0].set_title("Корреляция Пирсона", fontsize=16)
    ax[1].set_title("Корреляция Спирмана", fontsize=16)

    return fig


def scatter_plot(
    df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, hue_order: list, title: str
) -> matplotlib.figure.Figure:
    """
    Построение scatter plot
    Parameters
    ----------
    df
        дата фрейм с данными
    x_col
        признак по х
    y_col
        признак по у
    hue_col
        признак для группировки
    hue_order
        порядок группировки
    title
        название графика

    Returns
    -------
    график
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        hue_order=hue_order,
        alpha=0.7,
        palette="rocket",
    )
    sns.regplot(df, x=x_col, y=y_col, scatter=False, color="#701f57")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_col, fontsize=14)
    ax.set_ylabel(y_col, fontsize=14)

    return fig


def box_plot(
    df: pd.DataFrame, x_col: str, y_col: str, order: list, title: str
) -> matplotlib.figure.Figure:
    """
    Построение box plot
    Parameters
    ----------
    df
        дата фрейм с данными
    x_col
        признак по х
    y_col
        признак по у
    title
        название графика

    Returns
    -------
    график
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.boxplot(
        data=df,
        x=x_col,
        y=y_col,
        order=order,
        palette="rocket",
        hue=y_col,
        legend=False,
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_col, fontsize=14)
    ax.set_ylabel(y_col, fontsize=14)

    return fig


def line_plot(
    df: pd.DataFrame, x_col: str, y_col: str, title: str, mean: bool = False
) -> matplotlib.figure.Figure:
    """
     Построение линейного графика

     Parameters
     ----------
    df
         дата фрейм с данными
     x_col
         признак по х
     y_col
         признак по у
     title
         название графика
     mean
         нужно ли строить среднее

     Returns
     -------
     график
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.lineplot(data=df, x=x_col, y=y_col, color="#701f57")

    if mean:
        target_mean = df[y_col].mean()
        plt.axhline(target_mean, linestyle="--", color="#701f57")
        plt.text(y=target_mean * 1.1, x=1910, s=f"mean = {round(target_mean, 2)}")

    ax.set_title(title, fontsize=16)
    ax.set_ylabel(y_col, fontsize=14)
    ax.set_xlabel(x_col, fontsize=14)

    return fig
