import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd


STOP_WORDS = set(stopwords.words("english"))

NEGATION_WORDS = {"no", "not", "nor", "never"}
STOP_WORDS = STOP_WORDS - NEGATION_WORDS


LEMMATIZER = WordNetLemmatizer()

URL_PATTERN = r"http\S+|www\.\S+"
MENTION_PATTERN = r"@\w+"


def normalize_basic(text: str) -> str:
    text = text.lower()
    text = re.sub(URL_PATTERN, " ", text)
    text = re.sub(MENTION_PATTERN, " ", text)
    text = re.sub("rt", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    return nltk.word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    return [LEMMATIZER.lemmatize(t) for t in tokens]


# zzzzzzzzzzzzzzzzzzz


def preprocess_simple(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = normalize_basic(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


def preprocess_advanced(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = normalize_basic(text)
    tokens = tokenize_text(text)
    tokens = [t for t in tokens if len(t) >= 2]

    return " ".join(tokens)


def preprocess_bert(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = re.sub(URL_PATTERN, " ", text)
    text = re.sub(MENTION_PATTERN, " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess(text: str, mode: str = "simple") -> str:
    if mode == "simple":
        return preprocess_simple(text)
    elif mode == "advanced":
        return preprocess_advanced(text)
    elif mode == "bert":
        return preprocess_bert(text)
    else:
        raise ValueError(f"Error : {mode}")


def drop_short_texts(
    df: pd.DataFrame, text_column: str, min_len: int = 2
) -> pd.DataFrame:
    df = df.copy()

    lengths = df[text_column].fillna("").astype(str).str.split().apply(len)

    mask_keep = lengths >= min_len

    dropped = (~mask_keep).sum()
    total = len(df)
    print(
        f"[drop_short_texts] Colonne '{text_column}': "
        f"{dropped} lignes supprimées sur {total} "
        f"({dropped / total * 100:.4f}%). "
        f"Min len = {min_len}"
    )

    return df[mask_keep].copy()
