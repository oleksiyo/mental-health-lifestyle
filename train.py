import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from scipy.sparse import vstack


# =========================
# Configuration
# =========================
DATA_PATH = os.getenv("DATA_PATH", "data/mental_health.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "model.bin")
RANDOM_STATE = 42
TARGET_COL = "Has_Mental_Health_Issue"


# =========================
# Utilities
# =========================
def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from {path}")
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    print(f"[INFO] Dataset shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_full_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )

    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.25,  # 60 / 20 / 20
        random_state=RANDOM_STATE,
        stratify=df_full_train[TARGET_COL],
    )

    return df_train, df_val, df_test


def prepare_features(df_train, df_val, df_test):
    y_train = df_train[TARGET_COL].values
    y_val = df_val[TARGET_COL].values
    y_test = df_test[TARGET_COL].values

    df_train = df_train.drop(columns=[TARGET_COL]).fillna(0)
    df_val = df_val.drop(columns=[TARGET_COL]).fillna(0)
    df_test = df_test.drop(columns=[TARGET_COL]).fillna(0)

    dv = DictVectorizer(sparse=True)

    X_train = dv.fit_transform(df_train.to_dict(orient="records"))
    X_val = dv.transform(df_val.to_dict(orient="records"))
    X_test = dv.transform(df_test.to_dict(orient="records"))

    return X_train, X_val, X_test, y_train, y_val, y_test, dv


def evaluate(model, X, y_true) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


# =========================
# Main pipeline
# =========================
def main() -> None:
    np.random.seed(RANDOM_STATE)

    # Load & split
    df = load_data(DATA_PATH)
    df_train, df_val, df_test = split_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test, dv = prepare_features(
        df_train, df_val, df_test
    )

    print("[INFO] Train / Val / Test shapes:")
    print(X_train.shape, X_val.shape, X_test.shape)

    # =========================
    # Hyperparameter tuning
    # =========================
    print("\n[INFO] Hyperparameter tuning: Logistic Regression")

    lr_params = {
        "C": np.logspace(-3, 2, 20),
        "class_weight": [None, "balanced"],
    }

    lr_base = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    lr_search = RandomizedSearchCV(
        estimator=lr_base,
        param_distributions=lr_params,
        n_iter=20,
        cv=5,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    lr_search.fit(X_train, y_train)
    best_lr = lr_search.best_estimator_

    val_metrics = evaluate(best_lr, X_val, y_val)
    print("[RESULT] Validation metrics:", val_metrics)
    print("[INFO] Best params:", lr_search.best_params_)

    # =========================
    # Final retrain (train + val)
    # =========================
    print("\n[INFO] Final retrain on train + validation")

    X_train_full = vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    best_lr.fit(X_train_full, y_train_full)

    test_metrics = evaluate(best_lr, X_test, y_test)
    print("[FINAL TEST METRICS]", test_metrics)

    # =========================
    # Save artifact
    # =========================
    artifact = {
        "model": best_lr,
        "dict_vectorizer": dv,
        "best_params": lr_search.best_params_,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "random_state": RANDOM_STATE,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"[INFO] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
