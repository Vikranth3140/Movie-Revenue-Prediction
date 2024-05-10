import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# Loading our dataset
df = pd.read_csv('../revised datasets\output.csv')

le = LabelEncoder()

categorical_features = ['released','writer','rating','name', 'genre', 'director', 'star', 'country', 'company']

for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Our features and target
features = df[['released','writer','rating','name', 'genre', 'director', 'star', 'country', 'company', 'runtime', 'score', 'budget', 'year', 'votes']]
target = df['gross']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1]
}

# Implementing GridSearchCV 
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(loss='squared_error', random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='r2',
                           n_jobs=-1)

grid_search.fit(features, target)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best R^2 Score:", best_score)

best_model = GradientBoostingRegressor(loss='squared_error', random_state=42, **best_params)

best_model.fit(X_train, y_train)

train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

# R2 scores
train_accuracy = r2_score(y_train, train_predictions)
test_accuracy = r2_score(y_test, test_predictions)

print(f'\nFinal Training Accuracy: {train_accuracy*100:.2f}%')
print(f'Final Test Accuracy: {test_accuracy*100:.2f}%')


# Encode categorical features
def encode_features(df):
    le = LabelEncoder()
    categorical_features = ['released','writer','rating','name', 'genre', 'director', 'star', 'country', 'company']
    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])
    return df

# CLI for predicting movie revenue
def predict_revenue_with_range(model, input_df, tolerance=10):
    input_df = input_df.reindex(columns=features.columns, fill_value=0)
    input_df = encode_features(input_df)
    prediction = model.predict(input_df)
    lower_bound = prediction - tolerance
    upper_bound = prediction + tolerance
    return lower_bound[0], upper_bound[0]

def main():
    # Get user input
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

    # Load the trained model
    model = best_model  # Use the previously trained model

    # Predict revenue
    # predicted_revenue = predict_revenue(model, user_input)
    # print(f"Predicted revenue for the movie: ${predicted_revenue:.2f} million")

    lower_bound, upper_bound = predict_revenue_with_range(model, user_input)

    print(f"Predicted revenue range for the movie: ${lower_bound:.2f} million to ${upper_bound:.2f} million")

if __name__ == "__main__":
    main()