import streamlit as st
import pandas as pd
from models.gradient_boost import best_model, le

# Function to preprocess the input
def preprocess_input(released, writer, rating, name, genre, director, star, country, company, runtime, score, budget, year, votes):
    released_encoded = le.fit_transform([released])
    writer_encoded = le.fit_transform([writer])
    rating_encoded = le.fit_transform([rating])
    name_encoded = le.fit_transform([name])
    genre_encoded = le.fit_transform([genre])
    director_encoded = le.fit_transform([director])
    star_encoded = le.fit_transform([star])
    country_encoded = le.fit_transform([country])
    company_encoded = le.fit_transform([company])

    input_data = pd.DataFrame({
        'released': released_encoded,
        'writer': writer_encoded,
        'rating': rating_encoded,
        'name': name_encoded,
        'genre': genre_encoded,
        'director': director_encoded,
        'star': star_encoded,
        'country': country_encoded,
        'company': company_encoded,
        'runtime': runtime,
        'score': score,
        'budget': budget,
        'year': year,
        'votes': votes,
    })

    return input_data

# Function to predict the gross range
def predict_gross_range(input_data):
    predicted_gross = best_model.predict(input_data)
    if predicted_gross <= 5000000:
        return "Low Revenue (<= $5M)"
    elif predicted_gross <= 25000000:
        return "Medium-Low Revenue ($5M - $25M)"
    elif predicted_gross <= 50000000:
        return "Medium Revenue ($25M - $50M)"
    elif predicted_gross <= 80000000:
        return "High Revenue ($50M - $80M)"
    else:
        return "Ultra High Revenue (>= $80M)"

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

with st.form(key='movie_form'):
    released = st.text_input("Release Date")
    writer = st.text_input("Writer")
    rating = st.selectbox("MPAA Rating", ['G', 'PG', 'PG-13', 'R', 'NC-17'])
    name = st.text_input("Movie Name")
    genre = st.text_input("Genre")
    director = st.text_input("Director")
    star = st.text_input("Leading Star")
    country = st.text_input("Country of Production")
    company = st.text_input("Production Company")
    runtime = st.number_input("Runtime (minutes)", min_value=0.0)
    score = st.number_input("IMDb Score", min_value=0.0, max_value=10.0)
    budget = st.number_input("Budget", min_value=0.0)
    year = st.number_input("Year of Release", min_value=1900, max_value=2100)
    votes = st.number_input("Initial Votes", min_value=0)
    submit_button = st.form_submit_button(label='Predict Revenue')

if submit_button:
    input_data = preprocess_input(released, writer, rating, name, genre, director, star, country, company, runtime, score, budget, year, votes)

    # Predict the revenue range
    predicted_gross_range = predict_gross_range(input_data)
    st.markdown('### Prediction Result')
    st.success(f'Predicted Revenue Range for "{name}": {predicted_gross_range}')