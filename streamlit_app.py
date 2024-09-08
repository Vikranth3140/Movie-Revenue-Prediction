import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def prepare_features(df):
    processed_df = preprocess_data(df)

    if "log_gross" in processed_df.columns:
        y = processed_df["log_gross"]
        X = processed_df.drop("log_gross", axis=1)
    else:
        y = None
        X = processed_df

    return X, y


def preprocess_data(df):
    df = df.copy()

    # Log Transformation
    if "gross" in df.columns:
        df["log_gross"] = np.log1p(df["gross"])
    df["log_budget"] = np.log1p(df["budget"])

    # Feature engineering
    df["budget_vote_ratio"] = df["budget"] / (df["votes"] + 1)
    df["budget_runtime_ratio"] = df["budget"] / (df["runtime"] + 1)
    df["budget_score_ratio"] = df["log_budget"] / (df["score"] + 1)
    df["vote_score_ratio"] = df["votes"] / (df["score"] + 1)
    df["budget_year_ratio"] = df["log_budget"] / (df["year"] - df["year"].min() + 1)
    df["vote_year_ratio"] = df["votes"] / (df["year"] - df["year"].min() + 1)
    df["score_runtime_ratio"] = df["score"] / (df["runtime"] + 1)
    df["budget_per_minute"] = df["budget"] / (df["runtime"] + 1)
    df["votes_per_year"] = df["votes"] / (df["year"] - df["year"].min() + 1)
    df["is_recent"] = (df["year"] >= df["year"].quantile(0.75)).astype(int)
    df["is_high_budget"] = (df["log_budget"] >= df["log_budget"].quantile(0.75)).astype(
        int
    )
    df["is_high_votes"] = (df["votes"] >= df["votes"].quantile(0.75)).astype(int)
    df["is_high_score"] = (df["score"] >= df["score"].quantile(0.75)).astype(int)

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
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    numerical_features = [
        "runtime",
        "score",
        "year",
        "votes",
        "log_budget",
        "budget_vote_ratio",
        "budget_runtime_ratio",
        "budget_score_ratio",
        "vote_score_ratio",
        "budget_year_ratio",
        "vote_year_ratio",
        "score_runtime_ratio",
        "budget_per_minute",
        "votes_per_year",
        "is_recent",
        "is_high_budget",
        "is_high_votes",
        "is_high_score",
    ]

    imputer = SimpleImputer(strategy="median")
    df[numerical_features] = imputer.fit_transform(df[numerical_features])

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    if "gross" in df.columns:
        df = df.drop(["gross", "budget"], axis=1)
    else:
        df = df.drop(["budget"], axis=1)

    return df


def run_model():
    df = pd.read_csv("revised datasets/output.csv")
    X, y = prepare_features(df)
    param_grid = {
        "n_estimators": [100, 500],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    }
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=42, **best_params
    )
    best_model.fit(X, y)
    return best_model

def predict_gross(input_data, best_model):
    processed_data = preprocess_data(pd.DataFrame([input_data]))
    expected_features = best_model.feature_names_in_
    for feature in expected_features:
        if feature not in processed_data.columns:
            processed_data[feature] = 0
    processed_data = processed_data[expected_features]
    log_prediction = best_model.predict(processed_data)
    prediction = np.exp(log_prediction) - 1
    return prediction[0]

def predict_gross_range(gross):
    if gross <= 10000000:
        return f"Low Revenue (<= $10M)"
    elif gross <= 40000000:
        return f"Medium-Low Revenue ($10M - $40M)"
    elif gross <= 70000000:
        return f"Medium Revenue ($40M - $70M)"
    elif gross <= 120000000:
        return f"Medium-High Revenue ($70M - $120M)"
    elif gross <= 200000000:
        return f"High Revenue ($120M - $200M)"
    else:
        return f"Ultra High Revenue (>= $200M)"

st.markdown(
    """
    <h1 style='text-align: center; color: cyan;'>Movie Revenue Prediction</h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h2 style='text-align: center; color: white;'>Movie Details</h2>
    """,
    unsafe_allow_html=True,
)

with st.form(key="movie_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        released = st.text_input("Release Date")
        writer = st.text_input("Writer")
        rating = st.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R", "NC-17"])
        name = st.text_input("Movie Name")
        genre = st.text_input("Genre")
        director = st.text_input("Director")
        star = st.text_input("Leading Star")
    
    with col2:
        country = st.text_input("Country of Production")
        company = st.text_input("Production Company")
        runtime = st.number_input("Runtime (minutes)", min_value=0.0)
        score = st.number_input("IMDb Score", min_value=0.0, max_value=10.0)
        budget = st.number_input("Budget", min_value=0.0)
        year = st.number_input("Year of Release", min_value=1900, max_value=2100)
        votes = st.number_input("Initial Votes", min_value=0)

    submit_button = st.form_submit_button(label="Predict Revenue")

if submit_button:
    input_data = {
        "released": released,
        "writer": writer,
        "rating": rating,
        "name": name,
        "genre": genre,
        "director": director,
        "star": star,
        "country": country,
        "company": company,
        "runtime": runtime,
        "score": score,
        "budget": budget,
        "year": year,
        "votes": votes,
    }
    
    best_model = run_model()
    predicted_gross = predict_gross(input_data, best_model)
    predicted_gross_range = predict_gross_range(predicted_gross)

    st.markdown("## Prediction Result")
    st.success(f'Predicted Revenue for "{name}": ${predicted_gross:,.2f}')
    st.success(f'Predicted Revenue Range: {predicted_gross_range}')