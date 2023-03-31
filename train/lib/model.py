import os
import pickle
from collections import defaultdict

import optuna
import pandas as pd
import mlflow

from prefect import task
from surprise import (
    Dataset,
    accuracy,
)
from surprise.model_selection import (
    cross_validate,
    LeaveOneOut,
)
from surprise.prediction_algorithms import (
    BaselineOnly,
    CoClustering,
    KNNBasic,
    KNNBaseline,
    KNNWithMeans,
    KNNWithZScore,
    NormalPredictor,
    SlopeOne,
    SVD,
    SVDpp,
)

from app_config import (
    ANIME_DATA,
    DATA_DIR,
    EXPERIMENT,
    N_TRIAL,
    OPTUNA_DB,
    PKL_MODEL_DIR,
    THRESHOLD,
)
from lib.utils import load_csv, save_model


def get_benchmark(
    data: Dataset, 
    lOO: bool = False
) -> pd.DataFrame:
    """
    Generates a benchmark DataFrame for a set of Surprise recommendation algorithms.

    Args:
    - data: A Surprise Dataset object containing the ratings data to use for training and testing.
    - lOO (bool): A boolean to configure the used of LeaveOneOut.

    Returns:
    - A pandas DataFrame containing the RMSE scores for each algorithm, along with their names.
    """
    
    experiment_full_name = f"{EXPERIMENT}_benchmark_loo.csv" if lOO else f"{EXPERIMENT}_benchmark.csv"
    cv = LeaveOneOut() if lOO else 5
    
    if os.path.exists(os.path.join(DATA_DIR, experiment_full_name)):
        # Load pre-existing benchmark DataFrame from CSV file
        with open(f"{EXPERIMENT}_benchmark.csv", "rb") as f:
            benchmark_df = load_csv(os.path.join(DATA_DIR, experiment_full_name))
            return benchmark_df
    else:
        benchmark = []
        # Iterate over all algorithms
        for algorithm in [NormalPredictor(), BaselineOnly(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), KNNBaseline(), SVD(), SVDpp(), SlopeOne(), CoClustering()]: 
            # Perform cross validation
            results = cross_validate(algorithm, data, measures=['RMSE'], cv=cv, verbose=False)

            # Get results & append algorithm name
            tmp = pd.DataFrame.from_dict(results).mean(axis=0)
            tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
            benchmark.append(tmp)
        benchmark_df = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')    
        
        # Save benchmark DataFrame to CSV file
        benchmark_df.to_csv(os.path.join(DATA_DIR, experiment_full_name), index = True)
        return benchmark_df


def calculate_metrics(
    predictions: list[tuple[int, int, float, float, any]],
    data: list[tuple[int, int, float]],
    threshold: float
) -> object:
    """
    Calculate various evaluation metrics for a given set of predictions.

    Args:
    - predictions (list[tuple[int, int, float, float, any]]): A list of tuples containing the user id, item id, true rating, estimated rating, and additional information (if any) for each prediction.
    - data (list[tuple[int, int, float]]): A list of tuples containing the user id, item id, and rating for each data point.
    - threshold (float): The threshold above which a rating is considered positive.

    Returns:
    - metrics (object): A dictionary containing the evaluation metrics.
    """
    # User coverage
    recommended_users = set()
    n_users = len(set([t[0] for t in data]))
    for uid, iid, true_r, est, _ in predictions:
        recommended_users.add(uid)

    # Initialize a dictionary to store the evaluation results
    metrics = defaultdict(float)

    # Define y_true and y_pred
    y_true = [true_r for uid, iid, true_r, est, _ in predictions]
    y_pred = [est if est >= threshold else threshold for uid, iid, true_r, est, _ in predictions]

    # Calculate various evaluation metrics
    metrics['hit_rate'] = len(set(y_true) & set(y_pred)) /  len(set(y_true)) if len(set(y_true)) != 0 else 1
    metrics['mae'] = accuracy.mae(predictions)
    metrics['rmse'] = accuracy.rmse(predictions)
    metrics['user_coverage'] = len(recommended_users) / n_users if n_users != 0 else 0

    return metrics


def get_hyperparams(
    trial: optuna.Trial,
    hyperparams_dict: object
) -> object:
    """
    Generate hyperparameters for Surprise model using Optuna.

    Args:
    - trial (optuna.Trial): The optuna trial object.
    - hyperparams_dict (dict): A dictionary containing the hyperparameters to optimize.

    Returns:
    - dict: A dictionary containing the optimized hyperparameters for the Surprise model.

    Raises:
    - ValueError: If the hyperparameter type is invalid.

    """
    hyperparams = {}
    for param, values in hyperparams_dict.items():
        if param == "bsl_options":
            hyperparams["bsl_options"] = {
                "method" : trial.suggest_categorical("method", ["als", "sgd"]),
                "n_epochs" : trial.suggest_int("n_epochs", 5, 20),
                "reg_u" : trial.suggest_float("reg_u", 5, 10),
                "reg_i" : trial.suggest_float("reg_i", 5, 10)
            }
        elif param == "sim_options":
            hyperparams["sim_options"] = {
                "name" : trial.suggest_categorical("name", ['cosine', 'msd', 'pearson']),
                "user_based" : True if trial.suggest_categorical("user_based", [True, False]) else False,
                "min_support" : trial.suggest_float("min_support", 1, 20)
            }
        elif values[0] == 'categorical':
            if param == 'user_based':
                hyperparams[param] = True if trial.suggest_categorical(param, [True, False]) else False
            else:
                hyperparams[param] = trial.suggest_categorical(param, values[1])
        elif values[0] == 'int':
            hyperparams[param] = trial.suggest_int(param, values[1], values[2])
        elif values[0] == 'uniform':
            hyperparams[param] = trial.suggest_uniform(param, values[1], values[2])
        elif values[0] == 'loguniform':
            hyperparams[param] = trial.suggest_loguniform(param, values[1], values[2])
        else:
            raise ValueError(f"Invalid parameter type: {values[0]}")
    return hyperparams


def objective(
    trial: any, 
    data: Dataset, 
    model: any,
    hyperparams: object,
    cv: int=5, 
    verbose: bool=False
) -> float:
    """
    Optimize a model with hyperparameters using cross-validation, log the results with mlflow, and return the 
    minimum RMSE value.
    
    Args:
    - trial: An object representing a single trial of an experiment with a hyperopt optimization algorithm.
    - data: A Surprise Dataset object containing the ratings data to use for training and testing.
    - model: A Surprise algorithm instance to use for training and testing.
    - hyperparams: A dictionary containing hyperparameters to optimize over.
    - cv: An optional integer indicating the number of cross-validation folds to use.
    - verbose: An optional boolean indicating whether to print out progress information during cross-validation.
    
    Returns:
    - A float representing the minimum RMSE value achieved during cross-validation.
    """
    # Use hyperopt, an Distributed Hyperparameter Optimization, to get the hyperparameters to use for the current trial
    options = get_hyperparams(trial, hyperparams) 
    # Use the selected hyperparameters to instantiate a new model object
    if options:
        model_instance = model(**options)
    else:
        model_instance = model()

    # Evaluate the model with cross-validation using the Surprise library
    results = cross_validate(model_instance, data, measures=['rmse'], cv=cv, verbose=verbose)
    # Get the minimum RMSE score from the cross-validation results
    rmse = results['test_rmse'].min()
    
    # Return the minimum RMSE value as the objective value for this trial
    return rmse


@task(name="Train model", tags=['Model'])
def fit_model(
    data: Dataset,
    name: str,
    model: any,
    hyperparams_dict: object=None,
    bsl_options: bool=False
) -> object:
    """
    Fit a surprise model with the given dataset and hyperparameters using either
    Optuna hyperparameter search or a default parameter setting.
    
    Args:
    - data (Dataset): Surprise dataset to use for training and evaluation.
    - name (str): A name to identify the model in the experiment.
    - model (any): The surprise model to use for training.
    - hyperparams_dict (object): Dictionary containing hyperparameters
      to be optimized with Optuna. If None, the default hyperparameters will be used.
    - bsl_options (bool): Whether to optimize hyperparameters for the SVD baseline algorithm.
      Default is False.
    
    Returns:
    - object: A dictionary containing the trained model, best hyperparameters
      (if hyperparams_dict is not None), and the RMSE score of the best model.
    """
    client = mlflow.MlflowClient()
    EXPERIMENT_NAME = f"{EXPERIMENT}-{name}"
    if os.path.exists(f"{PKL_MODEL_DIR}/{EXPERIMENT_NAME}.pkl"):
        with open(f"{PKL_MODEL_DIR}/{EXPERIMENT_NAME}.pkl", "rb") as f:
            model = pickle.load(f)
            return model
    else:
        print(f"Model {name} not exist yet.")
        # Set up the experiment
        mlflow.set_experiment(EXPERIMENT_NAME)

        # Fit the model
        with mlflow.start_run(run_name="fit_model", nested=True) as run:
            run_id = run.info.run_id
            if hyperparams_dict is None:
                # Train the model using default hyperparameters
                best_model = model().fit(data.build_full_trainset()) # fit on the whole train data set
                
                # Register model locally in pickle file
                save_model({'model': best_model}, EXPERIMENT_NAME)

                # Log model and experiment parameters
                mlflow.sklearn.log_model(best_model, "models")
                mlflow.register_model(f"runs:/{run_id}/models", EXPERIMENT_NAME)
                # mlflow.log_param('MAX_USER_RATING', MAX_USER_RATING)
                # mlflow.log_param('MAX_RATINGS_PER_ANIME', MAX_RATINGS_PER_ANIME)
                # mlflow.log_param('MIN_RATING', MIN_RATING)
                # Stage the model in production
                client.transition_model_version_stage(
                    name=EXPERIMENT_NAME, version=1, stage="Production"
                )

                # Return the trained model
                return {'model': best_model}
            else:
                # Set up Optuna study
                study = optuna.create_study(
                    storage=OPTUNA_DB,  
                    study_name=name,
                    direction='minimize',
                    load_if_exists=True)

                # Run the hyperparameter search
                study.optimize(lambda trial: objective(trial, data, model, hyperparams_dict), n_trials=N_TRIAL)

                # Get the best hyperparameters and the best score
                best_hyperparameters = study.best_params
                if bsl_options:
                    best_hyperparameters = {'bsl_options': {'method': best_hyperparameters['method'], 
                                                        'n_epochs': best_hyperparameters['n_epochs'], 
                                                        'reg_i': best_hyperparameters['reg_i'], 
                                                        'reg_u': best_hyperparameters['reg_u']}}
                fit_rmse = study.best_value
                best_model = model(**best_hyperparameters)

                # Fit the model with the best hyperparameters
                best_model.fit(data.build_full_trainset()) # fit on the whole train data set

                # Register model locally in pickle file
                save_model({'model': best_model, 'hyperparameters': best_hyperparameters, 'fit_rmse': fit_rmse}, EXPERIMENT_NAME)

                mlflow.sklearn.log_model(best_model, "models")
                # Register the model with mlflow
                mlflow.register_model(f"runs:/{run_id}/models", EXPERIMENT_NAME)
                # Log the model parameters and performance metrics with mlflow
                mlflow.log_params(best_hyperparameters)
                # mlflow.log_param('MAX_USER_RATING', MAX_USER_RATING)
                # mlflow.log_param('MAX_RATINGS_PER_ANIME', MAX_RATINGS_PER_ANIME)
                # mlflow.log_param('MIN_RATING', MIN_RATING)
                mlflow.log_metric("rmse", fit_rmse)
                # Stage the model in production
                client.transition_model_version_stage(
                    name=EXPERIMENT_NAME, version=1, stage="Production"
                )

                # Return the best model with the best hyperparameters and the score
                return {'model': best_model, 'hyperparameters': best_hyperparameters, 'fit_rmse': fit_rmse}


@task(name="Evaluate model", tags=['Model'])
def evaluate_model(
    name: str,
    model: any,
    trainset: list,
    testset: list,
    threshold: float=THRESHOLD
) -> object:
    """
    Evaluates a model on a given dataset and returns the evaluation metrics and predictions.

    Args:
    - name (str): The name of the model.
    - model (Trainset): The model to be evaluated.
    - trainset (list): The dataset to be used for biased evaluation.
    - testset (list): The dataset to be used for unbiased evaluation.
    - threshold (float): The threshold above which a rating is considered positive. Defaults to THRESHOLD.

    Returns:
    - dict: A dictionary containing the evaluation metrics and the predictions.
    """
    try:
        with mlflow.start_run(run_name="evaluate_model", nested=True) as run:
            run_id = run.info.run_id
            EXPERIMENT_NAME = f"{EXPERIMENT}-{name}"
            mlflow.set_experiment(EXPERIMENT_NAME)

            # Compute biased and unbiased predictions
            biased_predictions = model.test(trainset)
            unbiased_predictions = model.test(testset)

            # Calculate evaluation metrics for biased and unbiased predictions
            biased_metrics = calculate_metrics(biased_predictions, trainset, threshold)
            unbiased_metrics = calculate_metrics(unbiased_predictions, testset, threshold)

            # Log the model and the evaluation metrics to MLflow
            mlflow.sklearn.log_model(model, "models")
            mlflow.register_model(f"runs:/{run_id}/models", EXPERIMENT_NAME)

            for metric, value in biased_metrics.items():
                mlflow.log_metric(f'biased_{metric}', value)
            for metric, value in unbiased_metrics.items():
                mlflow.log_metric(f'unbiased{metric}', value)

            return {'biased_metrics': biased_metrics, 'unbiased_metrics': unbiased_metrics, 'unbiased_predictions': unbiased_predictions, 'biased_predictions': biased_predictions}
    except Exception as e:
        print(e)
        return None

