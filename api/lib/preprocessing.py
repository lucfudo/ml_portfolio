from typing import Any

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

from surprise import Dataset, Reader

from app_config import *


def _prepare_anime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the anime dataset for modeling.
    
    Args:
    - df (pd.DataFrame): the anime dataset to prepare
    
    Returns:
    - pd.DataFrame: the cleaned and processed anime dataset
    """
    if isinstance(df, pd.DataFrame):
        logger.info("Pre-processing anime data...")

        # Remove anime without rating
        df['rating'] = df['rating'].replace(-1, np.nan)
        # Remove rows with null values
        df = df.dropna()
        
        # Convert 'episodes' column to int
        df['episodes'] = df['episodes'].replace('Unknown', 0)
        df['episodes'] = df['episodes'].astype(int)
        
        # Replace genre by cluster
        # Replace missing values with an empty string
        df['genre'].fillna('', inplace=True)
        # Create a TfidfVectorizer object with stop words removed
        tfidf = TfidfVectorizer(stop_words='english')
        # Transform the genre column of the anime dataframe into a sparse matrix of TF-IDF features
        X = tfidf.fit_transform(df['genre'])
        # Apply K-means with 10 clusters
        kmeans = KMeans(n_clusters=10, random_state=0, n_init=10)
        kmeans.fit(X)
        # Retrieve cluster labels for each genre
        labels = kmeans.labels_
        # Change gere for cluster labels 
        df['genre'] = labels
        
        # Label encoded type
        label_encoder = LabelEncoder()
        df['type'] = label_encoder.fit_transform(df['type'])
    else:
        raise ValueError("Anime data must be a pandas dataframe")
    
    return df


def _prepare_rating(
    df_a: pd.DataFrame,
    df_r: pd.DataFrame,
    max_user_ratings: int=MAX_USER_RATING,
    max_ratings_per_anime: int=MAX_RATINGS_PER_ANIME
) -> pd.DataFrame:
    """
    This function prepares the ratings dataframe by removing non-ratings, 
    filtering anime that are not in the anime dataframe, and applying 
    thresholds for the number of ratings per user and anime.

    Args:
    - anime (pd.DataFrame): The anime dataframe.
    - rating (pd.DataFrame): The ratings dataframe.
    - max_user_ratings (int, optional): The maximum number of ratings per user allowed in the filtered DataFrame. Defaults to MAX_USER_RATING.
    - max_ratings_per_anime (int, optional): The maximum number of ratings per anime allowed in the filtered DataFrame. Defaults to MAX_RATINGS_PER_ANIME.

    Returns:
    - pd.DataFrame: The prepared ratings dataframe.
    """
    if isinstance(df_a, pd.DataFrame) and isinstance(df_r, pd.DataFrame):
        logger.info("Pre-processing rating data...")

        # Remove non rating
        df_r['rating'] = df_r['rating'].replace(-1, np.nan)
        df_r = df_r.dropna()
        
        # Remove anime that are not in the anime dataframe
        merged_df = df_a.merge(df_r, on='anime_id', how='inner')
        df_r = merged_df[['user_id', 'anime_id', 'rating_y']]
        df_r = df_r.rename(columns={'rating_y': 'rating'})
        
        # Apply thresholds for the number of ratings per user and anime
        user_counts = df_r.groupby('user_id')['rating'].count()
        user_counts = user_counts[user_counts <= max_user_ratings]
        
        anime_counts = df_r.groupby('anime_id').size()
        anime_counts = anime_counts[anime_counts <= max_ratings_per_anime]
        
        df_r = df_r[(df_r["user_id"].isin(user_counts.index)) & (df_r["anime_id"].isin(anime_counts.index))]
    else:
        raise ValueError("Anime data and rating data must be pandas dataframe")
        
    return df_r


def _get_rating_data(
    df: pd.DataFrame,
    min_rating: int = MIN_RATING
) -> Any:
    """
    This function prepares the rating dataset for use in the Surprise library 
    by modifying the columns names, filtering users and items with a minimum number 
    of ratings if specified, creating a Surprise Reader object, and splitting the 
    raw ratings into training and validation sets.

    Args:
    - df (pd.DataFrame): The original ratings dataframe.
    - min_rating (int): Filter users and items with a minimum number of ratings.

    """
    # Modify columns name
    df = df.rename(columns={"user_id": "user", "anime_id": "item"})
    
    # Filter the ratings data frame to limit to users who rated a minimum number of products
    user_counts = df["user"].value_counts()
    user_counts = user_counts[user_counts >= min_rating]
    df = df[df["user"].isin(user_counts.index)]
    
    # Filter the ratings data frame to limit to products that received a minimum number of ratings
    item_counts = df["item"].value_counts()
    item_counts = item_counts[item_counts >= min_rating]
    df = df[df["item"].isin(item_counts.index)]
    
    # Create Reader and load data
    reader = Reader(line_format="user item rating", sep=";", skip_lines=1)
    data = Dataset.load_from_df(df, reader)

    return data


def prepare_data(anime_df: pd.DataFrame, rating_df: pd.DataFrame, max_user_ratings: int = MAX_USER_RATING, 
                 max_ratings_per_anime: int = MAX_RATINGS_PER_ANIME, min_rating: int = MIN_RATING) -> Any:
    """
    Pre-processes and filters the input data to prepare it for model training or inference.

    Args:
    - anime_df (pd.DataFrame): The dataframe containing information about the anime.
    - rating_df (pd.DataFrame): The dataframe containing user ratings for the anime.
    - max_user_ratings (int): The maximum number of ratings per user to consider.
    - max_ratings_per_anime (int): The maximum number of ratings per anime to consider.
    - min_rating (int): The minimum rating to consider.

    Raises:
    - ValueError: If the input data is not in the expected format.

    """
    logger.info("Pre-processing data...")
    if not isinstance(anime_df, pd.DataFrame) or not isinstance(rating_df, pd.DataFrame):
        raise ValueError("Input data must be dataframes")
    
    anime_data = _prepare_anime(anime_df)
    rating_data = _prepare_rating(anime_data, rating_df, max_user_ratings=max_user_ratings, 
                                  max_ratings_per_anime=max_ratings_per_anime)
    data = _get_rating_data(rating_data, min_rating=min_rating)
    
    return data

