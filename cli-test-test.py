import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

# Loading our dataset
df = pd.read_csv('revised datasets/output.csv')

# Separate categorical and numerical features
categorical_features = ['released', 'writer', 'rating', 'name', 'genre', 'director', 'star', 'country', 'company']
numerical_features = ['runtime', 'score', 'budget', 'year', 'votes']

# Encode categorical features
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# Our features and target
features = df[categorical_features + numerical_features]
target = df['gross']

# Train the model on the entire dataset
model = GradientBoostingRegressor(loss='squared_error', random_state=42)
model.fit(features, target)

# Function to predict gross for a given movie
def predict_gross(movie_data):
    # Encode categorical features in the input data
    for feature, le in label_encoders.items():
        if feature in movie_data:
            movie_data[feature] = le.transform([movie_data[feature]])[0]
    
    # Predict the gross for the given movie data
    gross_prediction = model.predict([list(movie_data.values())])[0]
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