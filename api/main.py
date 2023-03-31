from typing import List, Tuple

import mlflow

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app_config import (
    logger,
    APP_TITLE,
    APP_DESCRIPTION,
    APP_VERSION,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_URI,
)
from lib.modelling import run_inference
from lib.utils import get_top_N

app = FastAPI(title=APP_TITLE,
              description=APP_DESCRIPTION,
              version=APP_VERSION)

# Add CORS middleware to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    user_id: int

class PredictionOut(BaseModel):
    top_n: List[Tuple[str, float, int]]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
pipeline = mlflow.sklearn.load_model(model_uri=REGISTERED_MODEL_URI)
anime_df, predictions = run_inference(pipeline)

@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData):
    logger.info(f"Getting top N for user...")
    try:
        top_n = get_top_N(anime_df, predictions, payload.user_id)
    except:
        logger.exception("Failed to get top N for user")
        return {"top_n": []}
    return {"top_n": top_n}