from collections import defaultdict

import joblib

import pandas as pd

from app_config import logger, TOP_N, THRESHOLD


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
    logger.info(f"Loading file from {path}")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.info(f"File from {path} not found")


def load_pipeline(path: str):
    logger.info(f"Loading pipeline from {path}")
    return joblib.load(path)


def get_top_N(
    anime: pd.DataFrame,
    predictions: list[tuple],
    user_id: int,
    n: int=TOP_N,
    threshold: float=THRESHOLD
) -> list[tuple[str, float, int]]:
    """
    Given a DataFrame of anime, a list of predictions, a user ID, and optional parameters for the number of recommendations
    to generate and the rating threshold, returns a list of top N anime recommendations for the specified user, along with 
    each recommended item's title, average score, and number of scores.
    
    Args:
    - anime: A DataFrame of anime data, including anime IDs, titles, ratings, and member counts.
    - predictions: A list of tuples representing predicted ratings for each user-item pair in the test set, in the format
                 (user_id, item_id, true_rating, estimated_rating, _).
    - user_id: An integer representing the user ID for which to generate recommendations.
    - n: An integer representing the number of recommendations to generate for the specified user. Defaults to TOP_N.
    - threshold: A float representing the minimum rating threshold for recommended items. Defaults to THRESHOLD.
    
    Returns:
        A list of tuples representing the top N recommended anime for the specified user, each in the format 
        (title, average_score, num_scores).
    """
    # Create a dictionary to store the top N recommendations for each user
    top_n = defaultdict(list)

    # Loop through each prediction in the test predictions
    for uid, iid, true_r, est in predictions:
        # Store the estimated rating for this item
        top_n[uid].append((iid, est))
    # Loop through each user in the dictionary and sort their recommendations by estimated rating
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        
    # Get the top N recommendations for the specified user
    top_n_for_user = [(iid, est) for (iid, est) in top_n[user_id] if anime[anime['anime_id'] == int(iid)]['rating'].values[0] >= threshold][:n]

    # Create a list of tuples containing the title, average score, and number of scores for each recommended item
    results = []
    for item_id, estimated_rating in top_n_for_user:
        item = anime[anime['anime_id'] == int(item_id)]
        title = item['name'].values[0]
        avg_score = item['rating'].values[0]
        num_scores = item['members'].values[0]
        results.append((title, avg_score, num_scores))
    
    # Return the list of recommended items with their titles, average scores, and number of scores
    return results
