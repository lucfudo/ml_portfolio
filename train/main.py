import mlflow

from prefect import flow
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import (
   CronSchedule,
   IntervalSchedule,
)

from app_config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_URI

from lib.model import fit_model, evaluate_model
from lib.preprocessing import prepare_data, prepare_anime, prepare_rating
from lib.utils import load_data

from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.knns import (KNNBasic, KNNBaseline,
                                                 KNNWithMeans, KNNWithZScore)
from surprise.prediction_algorithms.matrix_factorization import (SVD, SVDpp)
from surprise.prediction_algorithms.random_pred import NormalPredictor
from surprise.prediction_algorithms.slope_one import SlopeOne


@flow(name="Machine learning workflow", retries=1, retry_delay_seconds=30)
def complete_ml():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    anime_df, rating_df = load_data()

    anime_data = prepare_anime(anime_df)
    rating_data = prepare_rating(anime_data, rating_df)
        
    data, train_data, test_data = prepare_data(rating_data)

    fit_model(data, 'NormalPredictor', NormalPredictor)

    hyper_BaselineOnly = {
        "bsl_options": 0
    }
    fit_model(data, 'BaselineOnly', BaselineOnly, hyper_BaselineOnly, bsl_options=True)

    hyper_SVD = {
        "n_factors": ["int", 50, 200],
        "n_epochs": ["int", 10, 50],
        "lr_all": ["loguniform", 1e-4, 1e-1],
        "reg_all": ["loguniform", 1e-4, 1e-1],
    }
    fit_model(data, 'SVD', SVD, hyper_SVD)

    fit_model(data, 'SlopeOne', SlopeOne)

    hyper_CoClustering = {
        "n_cltr_u": ["int", 3, 20],
        "n_cltr_i": ["int", 3, 20],
        "n_epochs": ["int", 10, 50],
        "random_state": ["int", 0, 100],
    }
    fit_model(data, 'CoClustering', CoClustering, hyper_CoClustering)


@flow(name="Batch inference", retries=1, retry_delay_seconds=30)
def batch_inference():
    try:
        anime_data, rating_data = load_data()
    except FileNotFoundError as e:
        raise e

    data, train_data, test_data = prepare_data(rating_data)

    # Make predictions on the given dataset
    predictions = []
    for uid, iid, true_r, _ in data.raw_ratings:
        try:
            pipeline = mlflow.sklearn.load_model(model_uri=REGISTERED_MODEL_URI)
            pred = pipeline.predict(uid, iid, verbose=False)
            predictions.append((uid, iid, true_r, pred.est))
        except e:
            raise e

    return predictions


inference_deployment_every_minute = Deployment.build_from_flow(
    name="Model Inference Deployment",
    flow=batch_inference,
    version="1.0",
    tags=["inference"],
    schedule=IntervalSchedule(interval=600),
)

if __name__ == "__main__":
    # complete_ml()
    # inference = batch_inference()
    inference_deployment_every_minute.apply()
    flow.register(project_name="anime")
