from sklearn.pipeline import Pipeline

from surprise import PredictionImpossible

from lib.preprocessing import prepare_data
from lib.utils import load_csv

from app_config import logger, ANIME_DATA, RATING_DATA

def run_inference(pipeline: Pipeline) -> float:
    """
    Takes a pre-fitted pipeline (dictvectorizer + model)
    """
    logger.info("Running inference on payload...")
    anime_df = load_csv(ANIME_DATA)
    rating_df = load_csv(RATING_DATA)
    data = prepare_data(anime_df, rating_df)

    # Make predictions on the given dataset
    logger.info(f"Making predictions...")
    predictions = []
    for uid, iid, true_r, _ in data.raw_ratings:
        try:
            pred = pipeline.predict(uid, iid, verbose=False)
            predictions.append((uid, iid, true_r, pred.est))
        except PredictionImpossible:
            logger.info(f"Prediction impossible")
            pass
        
    return anime_df, predictions
