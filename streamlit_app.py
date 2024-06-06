import streamlit as st
import pandas as pd
import numpy as np
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
        return f"Low Revenue (<= $5M)"
    elif predicted_gross <= 25000000:
        return f"Medium-Low Revenue ($5M - $25M)"
    elif predicted_gross <= 50000000:
        return f"Medium Revenue ($25M - $50M)"
    elif predicted_gross <= 80000000:
        return f"High Revenue ($50M - $80M)"
    else:
        return f"Ultra High Revenue (>= $80M)"

st.title('Movie Revenue Prediction')

released = st.text_input("When is the movie going to be released")
writer = st.text_input("Who is the writer of the movie")
rating = st.selectbox("What is the MPAA rating of the movie", ['G', 'PG', 'PG-13', 'R', 'NC-17'])
name = st.text_input("What is the name of the movie")
genre = st.text_input("What is the genre of the movie")
director = st.text_input("Who is the director of the movie")
star = st.text_input("Who is the star of the movie")
country = st.text_input("Which country are you based")
company = st.text_input("Which company is producing the movie")
runtime = st.number_input("What is the runtime of the movie", min_value=0.0)
score = st.number_input("How are the critics rating the movie in the initial screening", min_value=0.0, max_value=10.0)
budget = st.number_input("What is the budget of the movie", min_value=0.0)
year = st.number_input("Which year is the movie shot", min_value=1900, max_value=2100)
votes = st.number_input("What is the total number of votes for the movie in the initial survey", min_value=0)

if st.button('Predict Revenue'):
    input_data = preprocess_input(released, writer, rating, name, genre, director, star, country, company, runtime, score, budget, year, votes)

    # Predict the revenue range
    predicted_gross_range = predict_gross_range(input_data)
    st.write(f'Predicted Revenue Range for "{name}": {predicted_gross_range}')