from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timedelta
import json
import threading
from .schemas import HealthOut, TweetIn, PredictionOut, FeedbackIn, FeedbackOut
from .model_loader import load_model, predict_sentiment, label_to_str


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    print("[main] Modèle initialisé")
    yield


app = FastAPI(
    title="AirParadis - Prédiction Sentiment - API",
    description="API pour la prédiction de sentiment sur les tweets (via TF-IDF + Régression Logistique).",
    version="1.0.0",
    lifespan=lifespan,
)


ROOT = Path(__file__).resolve().parents[1]
LOGS_PATH = ROOT / "logs"
LOGS_PATH.mkdir(exist_ok=True)

FEEDBACK_LOG_PATH = LOGS_PATH / "feedback.log"

wrong_predictions_buffer = []
buffer_lock = threading.Lock()

ALERT_THRESHOLD = 3
ALERT_WINDOW = timedelta(minutes=5)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""
@app.on_event("startup")
def startup_event():
    load_model()
    print("[main] Modèle initialisé au démarrage.")
"""


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok")


@app.post("/predict", response_model=PredictionOut)
def predict(request: TweetIn) -> PredictionOut:
    label, proba = predict_sentiment(request.text)
    label_str = label_to_str(label)

    return PredictionOut(
        label=label,
        label_str=label_str,
        proba=proba,
    )


@app.post("/feedback", response_model=FeedbackOut)
def feedback(request: FeedbackIn) -> FeedbackOut:
    if not request.is_correct:
        log_wrong_prediction(
            text=request.text,
            prediction=request.prediction,
            proba=request.proba,
        )

    return FeedbackOut(status="received")


def _append_feedback_log(entry: dict) -> None:
    entry_with_ts = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **entry,
    }
    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry_with_ts, ensure_ascii=False) + "\n")


def _check_and_update_alerts() -> None:
    now = datetime.utcnow()
    with buffer_lock:
        cutoff = now - ALERT_WINDOW
        while wrong_predictions_buffer and wrong_predictions_buffer[0] < cutoff:
            wrong_predictions_buffer.pop(0)

        wrong_predictions_buffer.append(now)

        if len(wrong_predictions_buffer) >= ALERT_THRESHOLD:
            alert_entry = {
                "type": "ALERT",
                "message": f"{len(wrong_predictions_buffer)} mauvaises prédictions sur les {ALERT_WINDOW.total_seconds() / 60:.0f} dernières minutes",
            }
            _append_feedback_log(alert_entry)
            print("[ALERT]", alert_entry["message"])


def log_wrong_prediction(
    text: str, prediction: int, proba: float | None = None
) -> None:
    entry = {
        "type": "WRONG_PREDICTION",
        "text": text,
        "prediction": prediction,
        "proba": proba,
    }
    _append_feedback_log(entry)
    _check_and_update_alerts()
