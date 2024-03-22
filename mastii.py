import pandas as pd

# Read the masti.csv file
df = pd.read_csv('masti.csv')

# Drop the specified columns
columns_to_drop = ['budget_x', 'gross_x', 'director_name', 'genres', 'movie_title', 'star', 'imdb_score']
df = df.drop(columns_to_drop, axis=1)

# Remove duplicates
df = df.drop_duplicates()

# Drop rows with incomplete data
df = df.dropna()

# Rename the columns
df = df.rename(columns={'budget_y': 'budget', 'gross_y': 'gross'})

# Write the final CSV to updated_masti.csv
df.to_csv('updated_masti.csv', index=False)