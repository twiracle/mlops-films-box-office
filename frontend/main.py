"""
Программа: Фронтенд сервиса для просмотра EDA, тренировки модели и получения предсказаний
Версия: 1.0
"""

import os

import yaml
import streamlit as st
from src.data.get_data import load_data, get_data
from src.plot.charts import (
    bar_plot,
    correlation_plot,
    box_plot,
    line_plot,
    scatter_plot,
)
from src.train.train import train_model
from src.evaluate.evaluate import evaluate_input, evaluate_from_file


CONFIG_PATH = "../config/config.yaml"


def main_page():
    """
    Начальная страница с описанием проекта
    """
    st.title("MLOps проект: предсказание кассовых сборов фильма 🎬🎥")
    st.write(
        """
        Задачей данного проекта является предсказание кассовых сборов фильмов. Данные были собраны с сайта Кинопоиск 
        с использованием собственного парсера. Были выбраны данные о фильмах США за все годы с первых 1000 страниц 
        сайта (на следующих страницах информации о кассовых сборах практически не было)."""
    )
    st.markdown(
        """
        ### Описание полей 
            - title - название фильма
            - rating - пользовательский рейтинг на Кинопоиске
            - production_year - год производства
            - genre - жанр
            - director - режиссер
            - age_rating - возрастной рейтинг
            - release_date - дата релиза
            - duration - продолжительность
            - cast - актеры
            - budget - бюджет
            - box_office_usa - кассовые сборы в США
            - box_office_world - кассовые сборы по всему миру
            - target: общий кассовый сбор (США + мир)
    """
    )


def eda():
    """
    Разведочный анализ данных
    """
    st.markdown("# Разведочный анализ данных")
    st.markdown("Более детальный EDA по этой теме - https://github.com/twiracle/film-box-office-prediction")

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data = get_data(path=config["preprocessing"]["data_path"])
    clean_data = get_data(path=config["preprocessing"]["clean_data_path"])
    st.write(data.head())

    st.sidebar.markdown("Выберите гипотезу:")

    genre_target = st.sidebar.checkbox(
        "Фильмы определенных жанров в среднем собирают больше денег"
    )
    budget_rating_target = st.sidebar.checkbox(
        "На кассовые сборы в основном влияют бюджет и рейтинг фильма"
    )
    box_office = st.sidebar.checkbox(
        "Если фильм собирает много денег в мировом прокате, то обычно он собираем много денег и в США"
    )
    age_rating_target = st.sidebar.checkbox(
        "Возрастной рейтинг влияет на кассовые сборы"
    )
    director_target = st.sidebar.checkbox(
        "Более известные режиссеры снимают фильмы, которые приносят в среднем больше денег"
    )
    year_target = st.sidebar.checkbox("Более новые фильмы приносят больше денег")
    cast_target = st.sidebar.checkbox(
        "Наличие известных актеров положительно влияет на сборы"
    )

    if genre_target:
        st.markdown(
            """#### Гипотеза 1. Фильмы определенных жанров в среднем собирают больше денег"""
        )

        st.pyplot(
            bar_plot(
                df=clean_data,
                x_col="main_genre",
                y_col="target",
                title="Зависимость средних сборов от жанра",
            )
        )
        st.markdown(
            """
            ##### Выводы:
            - можно заметить, что в среднем больше всего приносят кассовые собры фильмов с такими жанрами: мультфильм, история и фантастика
            - хотя у исторических фильмов, в отличии от остальных жанров, довольно большой разброс значений
            - кроме того, можно выделить несколько жанров, которые в среднем приносят намного меньше денег, чем остальные: фильм-нуар, документальный, музыка и короткометражка. Это может быть связано с тем, что данные жанры больше ориентированы на узкий круг людей
            - также можно сказать, что, скорее всего, мультфильмы в среднем приносят больше денег из-за того, что на них обычно ходят всей семьей -> покупают больше билетов
            - гипотеза подтвердилась
            """
        )

    if budget_rating_target:
        st.markdown(
            """#### Гипотеза 2. На кассовые сборы в основном влияют бюджет и рейтинг фильма"""
        )

        st.pyplot(correlation_plot(df=clean_data))
        st.markdown(
            """
            ##### Выводы:
            - можно заметить, что при обоих видах корреляции рейтинг не так сильно влияет на сборы
            - все же больше влияет бюджет (особенно по корреляции Пирсона, хотя по Пирсону все же связь не такая сильная)
            - кроме того, длительность фильма влияет на кассовые сборы даже немного больше, чем рейтинг
            - но если анализировать всю тепловую карту, то все же сильных зависимостей нет
            - гипотеза подтвердилась лишь на половину
            """
        )

    if box_office:
        st.markdown(
            """#### Гипотеза 3. Если фильм собирает много денег в мировом прокате, то обычно он собираем много денег и в США"""
        )

        st.pyplot(
            scatter_plot(
                df=clean_data,
                x_col="box_office_usa",
                y_col="box_office_world",
                hue_col="age_rating",
                hue_order=["0+", "6+", "12+", "16+", "18+", "undefined"],
                title="Связь box_office_usa и box_office_world в зависимости от age_rating",
            )
        )
        st.markdown(
            """
            ##### Выводы:
            - действительно, фильмы, который собрали много в мировом прокате, собрали много и в США
            - кроме того, если рассматривать еще и возрастной рейтинг, то фильмы 12+ находятся в топе по сборам, но более подробно рассмотрим этот вопрос в одной из следующих гипотез
            - гипотеза подтвердилась
            """
        )

    if age_rating_target:
        st.markdown("""#### Гипотеза 4. Возрастной рейтинг влияет на кассовые сборы""")

        st.pyplot(
            box_plot(
                df=clean_data,
                y_col="age_rating",
                x_col="target",
                order=["0+", "6+", "12+", "16+", "18+", "undefined"],
                title="Зависимость age_rating и target",
            )
        )
        st.markdown(
            """
            ##### Выводы:
            - можно заметить, что возрастной рейтинг и правда влияет на кассовые сборы
            - в среднем больше денег собирают фильмы с возрастными категориями 0+, 6+ и 12+
            - как было сказано ранее, это может быть связано с тем, что такие фильмы охватывают более широкую аудиторию и на них могут ходить всей семьей
            - гипотеза подвердилась
            """
        )

    if director_target:
        st.markdown(
            """#### Гипотеза 5. Более известные режиссеры снимают фильмы, которые приносят в среднем больше денег"""
        )

        st.pyplot(
            line_plot(
                df=clean_data,
                x_col="director_film_count",
                y_col="target",
                title="Зависимость известности режиссера и сборов фильма",
            )
        )
        st.markdown(
            """
            ##### Выводы:
            - если считать, что чем больше фильмов снял режиссер, тем он известнее, то вырисовывается такая картина
            - в целом, можно заметить, что режиссеры с большим количеством фильмов снимают фильмы, которые в среднем приносят больше, особенно если сравнивать с режиссерами, у которых по одному фильму
            - но если посмотреть на режиссера с наибольшим количеством фильмов, то его фильмы в среднем собрали не так уж и много - примерно на одном уровне с режиссерами, у которых по одному фильму
            - хотя несколько предыдущих режиссеров с большим количеством фильмов все же собирали довольно много денег
            - гипотеза подтвердилась, если отталкиваться от такого обозначения известности
            """
        )

    if year_target:
        st.markdown("""#### Гипотеза 6. Более новые фильмы приносят больше денег""")

        st.pyplot(
            line_plot(
                df=clean_data,
                x_col="production_year",
                y_col="target",
                title="Зависимость production_year и target",
                mean=True,
            )
        )
        st.markdown(
            """
            ##### Выводы:
            - действительно, обычно чем новее фильм, тем больше денег он принес - фильмы от 1995 года в среднем приносят больше денег, чем среднее значение выручки
            - единственный существенный спад был в 2020 - скорее всего, из-за пандемии и карантинов люди все же сидели дома, а не ходили в кино
            - гипотеза подтвердилась
            """
        )

    if cast_target:
        st.markdown(
            """#### Гипотеза 7. Наличие известных актеров положительно влияет на сборы"""
        )

        st.pyplot(
            line_plot(
                df=clean_data,
                x_col="actors_fame",
                y_col="target",
                title="Зависимость известности актеров и сборов фильма",
            )
        )
        st.markdown(
            """
            ##### Выводы:
            - если считать, что известность актеров соответсвует количеству фильмов, в которых они снялись, то можно сделать такие выводы
            - как видим, фильмы с более известными актерами и правда в среднем собирают больше денег, чем фильмы с не очень известными актерами
            - но в то же время, чем больше слава, тем больше разброс значений target - фильм может собрать и очень много, и очень мало
            - гипотеза не подствердилась, хоть и некая положительная зависимость есть
            """
        )


def training():
    """
    Тренировка модели
    """
    st.markdown("# Обучение LightGBM")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train_endpoint = config["endpoints"]["training"]

    if st.button("Обучить модель"):
        train_model(config, train_endpoint)


def prediction_from_file():
    """
    Получение предсказаний модели по файлу с данными
    """
    st.markdown("# Предсказание по файлу с данными")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    endpoint = config["endpoints"]["prediction_file"]

    upload_file = st.file_uploader("", type=["csv"], accept_multiple_files=False)

    if upload_file:
        dataset_csv_df, files = load_data(upload_file, "test")

        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(endpoint=endpoint, files=files)

        else:
            st.error("Сначала обучите модель")


def prediction_input():
    """
    Получение предсказаний модели по введенным данным
    """
    st.markdown("# Предсказание по введенным данным")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def main():
    """
    Сборка всех страниц
    """
    pages_names_dict = {
        "Описание проекта": main_page,
        "Разведочный анализ данных": eda,
        "Тренировка модели": training,
        "Предсказания по файлу": prediction_from_file,
        "Предсказания по вводу": prediction_input,
    }

    selected_page = st.sidebar.selectbox("Выберите страницу", pages_names_dict.keys())
    pages_names_dict[selected_page]()


if __name__ == "__main__":
    main()
