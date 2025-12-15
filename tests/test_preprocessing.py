from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PATH = ROOT / "scripts"
sys.path.append(str(SCRIPTS_PATH))

from preprocessing import preprocess_simple


def test_preprocess_simple_removes_url_and_mention():
    text = "@user Check this out http://example.com !!!"
    processed = preprocess_simple(text)

    assert "http" not in processed
    assert "@user" not in processed
    assert isinstance(processed, str)


def test_preprocess_simple_not_empty_on_normal_sentence():
    text = "I really love this airline, great service!"
    processed = preprocess_simple(text)

    assert isinstance(processed, str)
    assert processed != ""
