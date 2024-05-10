import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import numpy as np

# Load the trained model
def load_model():
    # Load the model here
    model = GradientBoostingRegressor(loss='squared_error', random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1)
    return model

# Encode categorical features
def encode_features(df):
    le = LabelEncoder()
    categorical_features = ['released','writer','rating','name', 'genre', 'director', 'star', 'country', 'company']
    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])
    return df

# Get user input for movie features
def get_input():
    name = input("Enter movie name: ")
    rating = input("Enter movie rating: ")
    genre = input("Enter movie genre: ")
    year = int(input("Enter movie year: "))
    released = input("Enter release date: ")
    score = float(input("Enter IMDb score: "))
    votes = int(input("Enter number of votes: "))
    director = input("Enter director's name: ")
    writer = input("Enter writer's name: ")
    star = input("Enter star cast: ")
    country = input("Enter country: ")
    budget = float(input("Enter budget in millions: "))
    company = input("Enter production company: ")
    runtime = int(input("Enter runtime in minutes: "))

    # Create a dataframe from user input
    user_input = pd.DataFrame({
        'name': [name],
        'rating': [rating],
        'genre': [genre],
        'year': [year],
        'released': [released],
        'score': [score],
        'votes': [votes],
        'director': [director],
        'writer': [writer],
        'star': [star],
        'country': [country],
        'budget': [budget],
        'company': [company],
        'runtime': [runtime]
    })
    return user_input

# Predict revenue for the input movie
def predict_revenue(model, input_df):
    input_df = encode_features(input_df)
    prediction = model.predict(input_df)
    return prediction[0]

# Main function
def main():
    # Load the trained model
    model = load_model()

    # Get user input
    user_input = get_input()

    # Predict revenue
    predicted_revenue = predict_revenue(model, user_input)

    print(f"Predicted revenue for the movie: ${predicted_revenue:.2f} million")

if __name__ == "__main__":
    main()