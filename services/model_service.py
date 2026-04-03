import json
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from services.text_preprocessing import preprocess

MODEL_PATH = "model.pkl"
VECTOR_PATH = "vector.pkl"
METRICS_PATH = "metrics.json"
MODEL_META_PATH = "model_meta.json"
MODEL_VERSION = "truthlens-passiveaggressive-v1"


def train_model():
    print("Training model...")

    true_df = pd.read_csv("datasets/True.csv")
    fake_df = pd.read_csv("datasets/Fake.csv")

    true_df["label"] = 0
    fake_df["label"] = 1
    data = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")
    data["content"] = data["content"].apply(preprocess)
    y = data["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data["content"],
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vector = TfidfVectorizer(max_features=5000)
    X_train = vector.fit_transform(X_train_text)
    X_test = vector.transform(X_test_text)

    model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_fake": round(float(precision_score(y_test, y_pred, pos_label=1)), 4),
        "recall_fake": round(float(recall_score(y_test, y_pred, pos_label=1)), 4),
        "f1_fake": round(float(f1_score(y_test, y_pred, pos_label=1)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(VECTOR_PATH, "wb") as vector_file:
        pickle.dump(vector, vector_file)
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    with open(MODEL_META_PATH, "w", encoding="utf-8") as meta_file:
        json.dump({"version": MODEL_VERSION}, meta_file, indent=2)

    print("Model trained and saved.")
    print("Evaluation metrics:", metrics)

    return model, vector, metrics


def load_or_train_model():
    if (
        os.path.exists(MODEL_PATH)
        and os.path.exists(VECTOR_PATH)
        and os.path.exists(MODEL_META_PATH)
    ):
        with open(MODEL_META_PATH, "r", encoding="utf-8") as meta_file:
            meta = json.load(meta_file)

        if meta.get("version") != MODEL_VERSION:
            return train_model()

        print("Loading existing model...")
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)
        with open(VECTOR_PATH, "rb") as vector_file:
            vector = pickle.load(vector_file)

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)
        else:
            metrics = None

        return model, vector, metrics

    return train_model()


model, vector, metrics = load_or_train_model()


def get_model_components():
    return model, vector, metrics
