import mlflow
import os

from prefect import Client, Flow, Task
from prefect.deployments import Deployment

from app_config import MLFLOW_TRACKING_URI

from lib.preprocessing import prepare_data
from lib.model import fit_model, evaluate_model
from lib.utils import load_data

from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.knns import (KNNBasic, KNNBaseline,
                                                 KNNWithMeans, KNNWithZScore)
from surprise.prediction_algorithms.matrix_factorization import (SVD, SVDpp)
from surprise.prediction_algorithms.random_pred import NormalPredictor
from surprise.prediction_algorithms.slope_one import SlopeOne


class MLFlowTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def run(self):
        return mlflow

class PrefectTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = Client('http://prefect:4200')

    def run(self):
        return self.client


with Flow("Machine learning workflow") as flow:
    anime_df, rating_df = load_data()
    data, train_data, test_data = prepare_data(anime_df, rating_df)
    mlflow = MLFlowTask()
    prefect = PrefectTask()
    fit_model(data, 'NormalPredictor', NormalPredictor)


if __name__ == "__main__":
    flow.deploy("train", deployment=Deployment("train"))


# import mlflow

# from prefect import flow
# from prefect.deployments import Deployment

# from app_config import MLFLOW_TRACKING_URI

# from lib.preprocessing import prepare_data
# from lib.model import fit_model, evaluate_model
# from lib.utils import load_data

# from surprise.prediction_algorithms.baseline_only import BaselineOnly
# from surprise.prediction_algorithms.co_clustering import CoClustering
# from surprise.prediction_algorithms.knns import (KNNBasic, KNNBaseline,
#                                                  KNNWithMeans, KNNWithZScore)
# from surprise.prediction_algorithms.matrix_factorization import (SVD, SVDpp)
# from surprise.prediction_algorithms.random_pred import NormalPredictor
# from surprise.prediction_algorithms.slope_one import SlopeOne


# @flow(name="Machine learning workflow", retries=1, retry_delay_seconds=30)
# def complete_ml():
#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#     anime_df, rating_df = load_data()
#     data, train_data, test_data = prepare_data(anime_df, rating_df)
#     fit_model(data, 'NormalPredictor', NormalPredictor)


# if __name__ == "__main__":
#     complete_ml()

# evaluate_model('NormalPredictor', best_model['model'], train_data, test_data, threshold=0)

# modeling_deployment_every_sunday = Deployment.build_from_flow(
#     name="Model training Deployment",
#     flow=complete_ml,
#     version="1.0",
#     tags=["model"],
#     schedule=CronSchedule(cron="0 0 * * 0")
# )


# if __name__ == "__main__":
#     complete_ml()
#     # modeling_deployment_every_sunday.apply()

# import prefect
# from prefect.run_configs.docker import DockerRun
# from prefect.storage.github import GitHub
# from prefect import Client

# # Define the flow
# flow = ...

# # Define the prefect client
# client = Client(api_key=os.environ.get("API_KEY"))

# # Register the flow
# flow_storage = GitHub(
#     repo="your/repo",
#     path="path/to/flow.py",
#     access_token_secret="GITHUB_ACCESS_TOKEN",
# )
# flow_id = flow.register(
#     project_name="project_name",
#     storage=flow_storage,
#     labels=["label1", "label2"],
# )

# # Define the docker run configuration
# run_config = DockerRun(
#     image="your_train_image_name",
#     labels=["label1", "label2"],
#     env={
#         "MLFLOW_TRACKING_URI": "http://mlflow:5000",
#         "EXPERIMENT": "test_w_min_rating",
#     },
# )

# # Trigger the flow
# flow_run_id = client.create_flow_run(
#     flow_id=flow_id,
#     run_config=run_config,
#     labels=["label1", "label2"],
# )