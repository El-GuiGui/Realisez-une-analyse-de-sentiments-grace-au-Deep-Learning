from pydantic import BaseModel
from datetime import datetime


class HealthOut(BaseModel):
    status: str


class TweetIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    label: int  # 0 ou 1
    label_str: str  # "negative" ou "positive"
    proba: float  # % proba de la classe positive


class FeedbackIn(BaseModel):
    text: str
    prediction: int
    proba: float | None = None
    is_correct: bool


class FeedbackOut(BaseModel):
    status: str


class StatsOut(BaseModel):
    total_predictions: int
    total_wrong_predictions: int
    error_rate: float


class WrongFeedbackOut(BaseModel):
    text: str
    predicted_label: int
    proba: float
    timestamp: datetime
