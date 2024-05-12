import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import numpy as np

# Loading our dataset
df = pd.read_csv('revised datasets/output.csv')

le = LabelEncoder()

categorical_features = ['released', 'writer', 'rating', 'name', 'genre', 'director', 'star', 'country', 'company']

for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Our features and target
features = df[['released', 'writer', 'rating', 'name', 'genre', 'director', 'star', 'country', 'company', 'runtime', 'score', 'budget', 'year', 'votes']]

target = df['gross']

# Train the model on the entire dataset
model = GradientBoostingRegressor(loss='squared_error', random_state=42)
model.fit(features, target)

# Function to predict gross for a given movie
def predict_gross(movie_data):
    # Transform the input data using LabelEncoder
    for feature in categorical_features:
        movie_data[feature] = le.transform([movie_data[feature]])[0]
    
    # Predict the gross for the given movie data
    gross_prediction = model.predict([movie_data.values.tolist()])[0]
    return gross_prediction

# Get input from the user for testing data
while True:
    movie_data = {}
    for feature in features.columns:
        value = input(f"Enter the value for '{feature}': ")
        movie_data[feature] = value
    
    # Predict the gross for the input movie data
    gross_prediction = predict_gross(movie_data)
    print(f"Predicted gross for the movie: {gross_prediction}")
    
    # Ask if the user wants to continue testing
    continue_testing = input("Do you want to test another movie? (yes/no): ").lower()
    if continue_testing != 'yes':
        break