from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

import os
import time
import smtplib
from email.mime.text import MIMEText
from collections import deque

from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import threading
from .schemas import (
    HealthOut,
    TweetIn,
    PredictionOut,
    FeedbackIn,
    FeedbackOut,
    StatsOut,
    WrongFeedbackOut,
)
from .model_loader import load_model, predict_sentiment, label_to_str


from dotenv import load_dotenv

load_dotenv()


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

TOTAL_PREDICTIONS = 0
TOTAL_WRONG = 0

ERROR_TIMESTAMPS = deque()
ALERT_WINDOW_MINUTES = 5
ALERT_THRESHOLD = 3

WRONG_FEEDBACKS = deque(maxlen=100)


ALERT_EMAIL_ENABLED = os.getenv("ALERT_EMAIL_ENABLED", "True").lower() == "true"
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO")
ALERT_EMAIL_SMTP = os.getenv("ALERT_EMAIL_SMTP", "smtp.gmail.com")
ALERT_EMAIL_PORT = int(os.getenv("ALERT_EMAIL_PORT", "587"))
ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD")
ALERT_EMAIL_USER = os.getenv("ALERT_EMAIL_USER")


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
    global TOTAL_PREDICTIONS

    label, proba = predict_sentiment(request.text)
    label_str = label_to_str(label)

    TOTAL_PREDICTIONS += 1

    return PredictionOut(
        label=label,
        label_str=label_str,
        proba=proba,
        text=request.text,
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


def _check_and_update_alerts(now: datetime, last_text: str) -> None:
    with buffer_lock:
        cutoff = now - ALERT_WINDOW

        while wrong_predictions_buffer and wrong_predictions_buffer[0] < cutoff:
            wrong_predictions_buffer.pop(0)

        wrong_predictions_buffer.append(now)

        if len(wrong_predictions_buffer) >= ALERT_THRESHOLD:
            header = (
                f"{len(wrong_predictions_buffer)} mauvaises prédictions sur les "
                f"{ALERT_WINDOW.total_seconds() / 60:.0f} dernières minutes"
            )

            lines = [
                header,
                f"Horodatage (UTC) : {now.isoformat()}",
                "",
                "Dernières prédictions erronées :",
            ]

            recent = list(WRONG_FEEDBACKS)[-3:]
            recent = list(reversed(recent))

            for i, it in enumerate(recent, start=1):
                txt = it["text"]
                if len(txt) > 100:
                    txt = txt[:100] + "…"
                lines.append(
                    f"{i}. label={it['predicted_label']}, "
                    f"proba={it['proba']:.3f}, "
                    f'texte="{txt}"'
                )

            email_body = "\n".join(lines)

            alert_entry = {
                "type": "ALERT",
                "message": header,
                "last_text": last_text[:200],
            }
            _append_feedback_log(alert_entry)
            print("[ALERT]", header)

            send_alert_email(email_body)


def log_wrong_prediction(
    text: str, prediction: int, proba: float | None = None
) -> None:
    global TOTAL_WRONG

    TOTAL_WRONG += 1

    entry = {
        "type": "WRONG_PREDICTION",
        "text": text,
        "prediction": prediction,
        "proba": proba,
    }
    _append_feedback_log(entry)

    now = datetime.utcnow()
    WRONG_FEEDBACKS.append(
        {
            "text": text,
            "predicted_label": prediction,
            "proba": proba if proba is not None else 0.0,
            "timestamp": now,
        }
    )

    _check_and_update_alerts(now, text)


def send_alert_email(message: str):
    if not ALERT_EMAIL_ENABLED:
        return

    if not (ALERT_EMAIL_FROM and ALERT_EMAIL_TO and ALERT_EMAIL_PASSWORD):
        print("[alert_email] Config email incomplète, pas d'envoi")
        return

    msg = MIMEText(message)
    msg["Subject"] = "Alerte modèle - trop de prédictions erronées"
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO

    with smtplib.SMTP(ALERT_EMAIL_SMTP, ALERT_EMAIL_PORT) as smtp:
        smtp.starttls()
        if ALERT_EMAIL_USER:
            smtp.login(ALERT_EMAIL_USER, ALERT_EMAIL_PASSWORD)
        else:
            smtp.login(ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD)
        smtp.sendmail(ALERT_EMAIL_FROM, [ALERT_EMAIL_TO], msg.as_string())


@app.get("/stats", response_model=StatsOut)
def get_stats() -> StatsOut:
    if TOTAL_PREDICTIONS == 0:
        error_rate = 0.0
    else:
        error_rate = TOTAL_WRONG / TOTAL_PREDICTIONS

    return StatsOut(
        total_predictions=TOTAL_PREDICTIONS,
        total_wrong_predictions=TOTAL_WRONG,
        error_rate=error_rate,
    )


@app.get("/wrong_feedbacks", response_model=list[WrongFeedbackOut])
def get_wrong_feedbacks(limit: int = 20) -> list[WrongFeedbackOut]:
    """Renvoie les derniers feedbacks négatifs pour analyse."""
    items = list(WRONG_FEEDBACKS)[-limit:]
    items = list(reversed(items))

    return [
        WrongFeedbackOut(
            text=it["text"],
            predicted_label=it["predicted_label"],
            proba=it["proba"],
            timestamp=it["timestamp"],
        )
        for it in items
    ]
