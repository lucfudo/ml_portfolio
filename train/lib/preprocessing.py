# Data analysis
import pandas as pd
import numpy as np


# Sklearn algorithms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


# Surprise algorithms
from surprise import Dataset, Reader


# Config
from app_config import (USE_MINIMUM, MAX_USER_RATING, MAX_RATINGS_PER_ANIME, MIN_RATING)
import random

def _prepare_anime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the anime dataset for modeling.
    
    Args:
    - df (pd.DataFrame): the anime dataset to prepare
    
    Returns:
    - pd.DataFrame: the cleaned and processed anime dataset
    """
    if isinstance(df, pd.DataFrame):
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


def prepare_data(
    df: pd.DataFrame,
    use_minimum: bool=False
) -> tuple:
    """
    This function prepares the rating dataset for use in the Surprise library 
    by modifying the columns names, filtering users and items with a minimum number 
    of ratings if specified, creating a Surprise Reader object, and splitting the 
    raw ratings into training and validation sets.

    Args:
    - df (pd.DataFrame): The original ratings dataframe.
    - use_minimum (bool): Whether to filter users and items with a minimum number 
        of ratings. Default is False.

    Returns:
    - tuple: A tuple containing the prepared dataset, the training data, and 
        the validation data.

    """
    # Modify columns name
    df = df.rename(columns={"user_id": "user", "anime_id": "item"})
    
    if use_minimum:
        # Filter the ratings data frame to limit to users who rated a minimum number of products
        user_counts = df["user"].value_counts()
        user_counts = user_counts[user_counts >= MIN_RATING]
        df = df[df["user"].isin(user_counts.index)]
        
        # Filter the ratings data frame to limit to products that received a minimum number of ratings
        item_counts = df["item"].value_counts()
        item_counts = item_counts[item_counts >= MIN_RATING]
        df = df[df["item"].isin(item_counts.index)]
    
    # Create Reader and load data
    reader = Reader(line_format="user item rating", sep=";", skip_lines=1)
    data = Dataset.load_from_df(df, reader)
    
    # Split the raw ratings into training and validation sets
    raw_ratings = data.raw_ratings
    random.shuffle(raw_ratings)
    threshold = int(0.8 * len(raw_ratings))
    train_raw_ratings = raw_ratings[:threshold]
    test_raw_ratings = raw_ratings[threshold:]

    data.raw_ratings = train_raw_ratings # data is now training set

    train_data = data.build_full_trainset().build_testset() # trainset: biased set
    test_data = data.construct_testset(test_raw_ratings) # testset: unbiased set
    return data, train_data, test_data
