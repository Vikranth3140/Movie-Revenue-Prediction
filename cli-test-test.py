import pandas as pd
import numpy as np
from models.gradient_boost import best_model, le  # Importing directly from models.gradient_boost

# Define the function to preprocess input data
def preprocess_input(released, writer, rating, name, genre, director, star, country, company, runtime, score, budget, year, votes):
    # Transform categorical features using LabelEncoder
    released_encoded = le.fit_transform([released])
    writer_encoded = le.fit_transform([writer])
    rating_encoded = le.fit_transform([rating])
    name_encoded = le.fit_transform([name])
    genre_encoded = le.fit_transform([genre])
    director_encoded = le.fit_transform([director])
    star_encoded = le.fit_transform([star])
    country_encoded = le.fit_transform([country])
    company_encoded = le.fit_transform([company])

    # Create a DataFrame with the preprocessed input
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

# Function to predict the gross
# def predict_gross(input_data):
#     return best_model.predict(input_data)

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
        return "Medium Revenue ($50M - $80M)"
    else:
        return "High Revenue ($25M - $50M)"

# Example usage
if __name__ == "__main__":
    # Take user input
    released = input("Enter the release date: ")
    writer = input("Enter the writer's name: ")
    rating = input("Enter the rating: ")
    name = input("Enter the movie name: ")
    genre = input("Enter the genre: ")
    director = input("Enter the director's name: ")
    star = input("Enter the star's name: ")
    country = input("Enter the country: ")
    company = input("Enter the production company: ")
    runtime = float(input("Enter the runtime in minutes: "))
    score = float(input("Enter the score: "))
    budget = float(input("Enter the budget: "))
    year = int(input("Enter the year: "))
    votes = float(input("Enter the number of votes: "))

    # Preprocess the input data
    input_data = preprocess_input(released, writer, rating, name, genre, director, star, country, company, runtime, score, budget, year, votes)

    # # Predict the gross
    predicted_gross_range = predict_gross_range(input_data)
    print(f'Predicted Revenue Range for "{name}": {predicted_gross_range}')
    # predicted_gross = predict_gross(input_data)
    # print(f'Predicted Gross for "{name}": ${predicted_gross[0]:,.2f}')