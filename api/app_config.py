import os

import logging

import pandas as pd


def get_logger(logging_level=logging.INFO, logger_name: str = "app_logger"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# LOGGING
LOGGER_LEVEL = "INFO"
logger = get_logger(logging_level=getattr(logging, LOGGER_LEVEL))


# File paths and directories
DATA_DIR = "/app/data"
ANIME_DATA = os.path.join(DATA_DIR, "anime.csv")
RATING_DATA = os.path.join(DATA_DIR, "rating.csv")

# MODELS
TOP_N = 10
THRESHOLD = 6
MAX_USER_RATING = 5000
MAX_RATINGS_PER_ANIME = 5000
MIN_RATING = 1

# MLFLOW
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT = "test_w_min_rating" # _wo_min_rating
NAME = "NormalPredictor"
REGISTERED_MODEL_NAME = f"{EXPERIMENT}-{NAME}"
STAGE = "production"
REGISTERED_MODEL_URI = f"models://mlflow/{REGISTERED_MODEL_NAME}/{STAGE}"


# MISC
APP_TITLE = "Anime"
APP_DESCRIPTION = ("A simple API to predict top N of anime for a specific user.")
APP_VERSION = "0.0.1"
# silence pandas `SettingWithCopyWarning` warnings
pd.options.mode.chained_assignment = None  # default='warn'
