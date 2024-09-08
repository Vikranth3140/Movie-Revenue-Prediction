import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from models.feature_scaling import preprocess_data, prepare_features

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

st.set_page_config(page_title="Movie Revenue Prediction System", layout="wide")

st.title("Movie Revenue Prediction System")

st.markdown("## Enter Movie Details")

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