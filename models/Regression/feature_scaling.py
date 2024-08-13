# Transformation of budget and reveneue to logarithmic scale for reducing our current model's overprediction tendencies.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_data(df):
    df = df.copy()

    if "gross" in df.columns:
        df["log_gross"] = np.log1p(df["gross"])

    df["log_budget"] = np.log1p(df["budget"])
    df["budget_vote_ratio"] = df["budget"] / (
        df["votes"] + 1
    )  # Adding 1 to avoid division by zero
    df["budget_runtime_ratio"] = df["budget"] / (df["runtime"] + 1)

    categorical_features = [
        "released",
        "writer",
        "rating",
        "name",
        "genre",
        "director",
        "star",
        "country",
        "company",
    ]

    for feature in categorical_features:
        df[feature] = df[feature].astype(str)

        value_counts = df[feature].value_counts()

        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    numerical_features = ["runtime", "score", "year", "votes"]

    imputer = SimpleImputer(strategy="median")
    df[numerical_features] = imputer.fit_transform(df[numerical_features])

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    df["budget_year"] = df["log_budget"] * df["year"]
    df["budget_score"] = df["log_budget"] * df["score"]

    if "gross" in df.columns:
        df = df.drop(["gross", "budget"], axis=1)
    else:
        df = df.drop(["budget"], axis=1)

    return df


def prepare_features(df):
    processed_df = preprocess_data(df)

    if "log_gross" in processed_df.columns:
        y = processed_df["log_gross"]
        X = processed_df.drop("log_gross", axis=1)
    else:
        y = None
        X = processed_df

    return X, y
