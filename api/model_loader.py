from pathlib import Path
from typing import Tuple
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PATH = ROOT / "scripts"
MODELS_PATH = ROOT / "models"

sys.path.append(str(SCRIPTS_PATH))

from preprocessing import preprocess_simple

MODEL_PATH = MODELS_PATH / "tfidf_logreg.joblib"

_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle introuvable à l'emplacement : {MODEL_PATH}"
            )
        _model = joblib.load(MODEL_PATH)
        print(f"[model_loader] Modèle chargé depuis {MODEL_PATH}")
    return _model


def predict_sentiment(text: str) -> Tuple[int, float]:
    model = load_model()

    text_clean = preprocess_simple(text)

    proba_pos = model.predict_proba([text_clean])[0][1]
    label = int(proba_pos >= 0.5)

    return label, float(proba_pos)


def label_to_str(label: int) -> str:
    return "negative" if label == 0 else "positive"


if __name__ == "__main__":
    exemple = "I love this airline, great service!"
    lab, proba = predict_sentiment(exemple)
    print("Texte :", exemple)
    print("Label :", lab, "(", label_to_str(lab), ")", "Proba classe 1 :", proba)
