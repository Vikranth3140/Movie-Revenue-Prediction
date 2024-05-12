import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

# Loading our dataset
df = pd.read_csv('revised datasets/output.csv')

# Different LabelEncoders for categorical features
le_released = LabelEncoder()
le_writer = LabelEncoder()
le_rating = LabelEncoder()
le_name = LabelEncoder()
le_genre = LabelEncoder()
le_director = LabelEncoder()
le_star = LabelEncoder()
le_country = LabelEncoder()
le_company = LabelEncoder()

# Apply LabelEncoding to categorical features
df['released'] = le_released.fit_transform(df['released'])
df['writer'] = le_writer.fit_transform(df['writer'])
df['rating'] = le_rating.fit_transform(df['rating'])
df['name'] = le_name.fit_transform(df['name'])
df['genre'] = le_genre.fit_transform(df['genre'])
df['director'] = le_director.fit_transform(df['director'])
df['star'] = le_star.fit_transform(df['star'])
df['country'] = le_country.fit_transform(df['country'])
df['company'] = le_company.fit_transform(df['company'])

# Our features and target
features = df[['released', 'writer', 'rating', 'name', 'genre', 'director', 'star', 'country', 'company', 'runtime', 'score', 'budget', 'year', 'votes']]
target = df['gross']

# Train the model on the entire dataset
model = GradientBoostingRegressor(loss='squared_error', random_state=42)
model.fit(features, target)

# Function to predict gross for a given movie
def predict_gross(movie_data):
    # Transform categorical features using the corresponding LabelEncoders
    movie_data['released'] = le_released.transform([movie_data['released']])[0]
    movie_data['writer'] = le_writer.transform([movie_data['writer']])[0]
    movie_data['rating'] = le_rating.transform([movie_data['rating']])[0]
    movie_data['name'] = le_name.transform([movie_data['name']])[0]
    movie_data['genre'] = le_genre.transform([movie_data['genre']])[0]
    movie_data['director'] = le_director.transform([movie_data['director']])[0]
    movie_data['star'] = le_star.transform([movie_data['star']])[0]
    movie_data['country'] = le_country.transform([movie_data['country']])[0]
    movie_data['company'] = le_company.transform([movie_data['company']])[0]

    # Predict the gross for the given movie data
    gross_prediction = model.predict([list(movie_data.values())])[0]
    return gross_prediction

# Sample movie data for prediction
movie_data = {
    'released': 'June 13, 1980 (United States)',
    'writer': 'Stephen King',
    'rating': 'R',
    'name': 'The Shining',
    'genre': 'Drama',
    'director': 'Stanley Kubrick',
    'star': 'Jack Nicholson',
    'country': 'United Kingdom',
    'company': 'Warner Bros.',
    'runtime': 146,
    'score': 8.4,
    'budget': 19000000,
    'year': 1980,
    'votes': 927000
}

# Predict the gross for the sample movie data
gross_prediction = predict_gross(movie_data)
print(f"Predicted gross for the movie: {gross_prediction}")