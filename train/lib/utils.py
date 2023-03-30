import os
import pickle
import pandas as pd
from prefect import task
from app_config import ANIME_DATA, RATING_DATA, PKL_MODEL_DIR


@task(name="Load csv", tags=['Serialize'])
def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file as a pandas DataFrame.

    Args:
    - path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the CSV data.

    Raises:
    - FileNotFoundError: If the file is not found at the specified path.

    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print("File not found")
    
       
@task(name="Load data", tags=['Serialize'])
def load_data():
    anime_df = load_csv(ANIME_DATA)
    rating_df = load_csv(RATING_DATA)
    return anime_df, rating_df


@task(name="Save model", tags=['Serialize'])
def save_model(model: object, name: str) -> None:
    """Saves a trained model to a file using pickle.

    Args:
    - model (object): A trained machine learning model object to be saved.
    - name (str): The name of the file to save the model.

    Returns:
    - None

    """
    if not os.path.exists(PKL_MODEL_DIR):
        os.makedirs(PKL_MODEL_DIR)
    with open(f"{PKL_MODEL_DIR}/{name}.pkl", "wb") as f:
        # Using pickle to dump the model object to a binary file
        pickle.dump(model, f)


@task(name="Load model", tags=['Serialize'])
def load_model(name: str) -> object:
    """Loads a trained model from a file using pickle.

    Args:
    - name (str): The name of the file containing the saved model.

    Returns:
    - object: A trained machine learning model object.

    """
    with open(f"{PKL_MODEL_DIR}/{name}.pkl", "rb") as f:
        # Using pickle to load the model object from the binary file
        model = pickle.load(f)
    return model

