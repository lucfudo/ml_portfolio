import os


# Default values and thresholds
NB_CLUSTER = 10
THRESHOLD = 6
MAX_USER_RATING = 500
MAX_RATINGS_PER_ANIME = 5000
MIN_RATING = 100
USE_MINIMUM = True


# File paths and directories
DATA_DIR = "/app/data"
ANIME_DATA = os.path.join(DATA_DIR, "anime.csv")
RATING_DATA = os.path.join(DATA_DIR, "rating.csv")
PKL_MODEL_DIR = os.path.join(DATA_DIR, "pkl_model")


# MLflow
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT = "test_w_min_rating"


# Optuna
OPTUNA_DB = "sqlite:///db.sqlite3"
N_TRIAL = 1
