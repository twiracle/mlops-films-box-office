"""
Программа: Конвейер для тренировки модели
Версия: 1.0
"""

import joblib
import yaml

from ..data.split_data import split_train_val_test
from ..train.train import find_best_params, train_model
from ..data.get_data import get_data
from ..transform.transform import pipeline_preprocessing


def pipeline_training(config_path: str) -> None:
    """
    Пайплайн для получения, предварительной обработки данных и обучения модели
    Parameters
    ----------
    config_path
        путь к конфигу

    Returns
    -------
    None
    """
    with open(config_path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config_preproc = config["preprocessing"]
    config_train = config["train"]

    df = get_data(config_preproc["data_path"])
    df = pipeline_preprocessing(df, flg_evaluation=False, **config_preproc)

    df_train, df_val, df_test = split_train_val_test(df, **config_preproc)
    study = find_best_params(df_train, df_val, df_test, **config_train)

    model = train_model(
        df_train=df_train, df_val=df_val, df_test=df_test, study=study, **config_train
    )

    joblib.dump(model, config_train["model_path"])
    joblib.dump(study, config_train["study_path"])
