from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "api"
sys.path.append(str(API_PATH))

from model_loader import load_model, predict_sentiment, label_to_str


def test_load_model_returns_sklearn_pipeline():
    model = load_model()
    assert hasattr(model, "predict_proba")


def test_predict_sentiment_output_types_and_ranges():
    text = "I love this airline, it was amazing!"
    label, proba = predict_sentiment(text)

    assert label in (0, 1)
    assert isinstance(proba, float)
    assert 0.0 <= proba <= 1.0

    label_str = label_to_str(label)
    assert label_str in ("negative", "positive")
