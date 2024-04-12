import pandas as pd
import matplotlib.pyplot as plt

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


# Plot the Gross vs Budget to see the variation and difference

# Load the dataset
df = pd.read_csv('new_updated_less-than-350m-dataset.csv')

# Plot gross vs budget
plt.figure(figsize=(10, 6))
plt.scatter(df['budget'], df['gross'], color='blue', alpha=0.5)
plt.title('Gross vs Budget')
plt.xlabel('Budget')
plt.ylabel('Gross')
plt.grid(True)
plt.show()


# Removed the movies from the budget where budget > 1 billion


# Read oldd_dataset.csv
df = pd.read_csv('oldd_dataset.csv')

# Remove items with budget more than 1000000000
df = df[df['budget'] <= 1000000000]

# Write the updated dataframe to bew_updated_less-than-1b-dataset.csv
df.to_csv('new_updated_less-than-1b-dataset.csv', index=False)



# Removed the movies from the budget where budget > 350 million


# Read oldd_dataset.csv
df = pd.read_csv('oldd_dataset.csv')

# Remove items with budget more than 350000000
df = df[df['budget'] <= 350000000]

# Write the updated dataframe to bew_updated_less-than-1b-dataset.csv
df.to_csv('new_updated_less-than-350m-dataset.csv', index=False)