from typing import List, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from surprise import PredictionImpossible

from lib.preprocessing import prepare_data
from lib.utils import load_csv
from app_config import logger, ANIME_DATA, RATING_DATA



def run_inference(pipeline: Pipeline, anime_data_path: str=ANIME_DATA, rating_data_path: str=RATING_DATA) -> Tuple[pd.DataFrame, List[Tuple[str, str, float, float]]]:
    """
    Runs inference on pre-fitted pipeline.

    Args:
    - pipeline (surprise.prediction_algorithms.algo_base.AlgoBase): A pre-fitted pipeline
      consisting of dictvectorizer and model.
    - anime_data_path (str): Path to the anime data CSV file.
    - rating_data_path (str): Path to the rating data CSV file.

    Returns:
    - Tuple[pd.DataFrame, List[Tuple[str, str, float, float]]]: A tuple containing pandas DataFrame
      of anime data and a list of tuples containing predictions.

    Raises:
    - FileNotFoundError: If the CSV file is not found at the specified path.
    """
    """
    Takes a pre-fitted pipeline (dictvectorizer + model)
    """
    try:
        logger.info("Loading CSV data...")
        anime_df = load_csv(anime_data_path)
        rating_df = load_csv(rating_data_path)
    except FileNotFoundError as e:
        logger.error(f"Error loading CSV data: {e}")
        raise e

    data = prepare_data(anime_df, rating_df)

    # Make predictions on the given dataset
    logger.info(f"Making predictions...")
    predictions = []
    for uid, iid, true_r, _ in data.raw_ratings:
        try:
            pred = pipeline.predict(uid, iid, verbose=False)
            predictions.append((uid, iid, true_r, pred.est))
        except PredictionImpossible:
            logger.info(f"Prediction impossible for user {uid} and item {iid}")
            pass

    return anime_df, predictions
