import pandas as pd

# Read the CSV file
df = pd.read_csv('movies.csv')


# Check the number of null values for 'budget' column
null_budget_count = df['gross'].isnull().sum()
print(f"Number of null values for 'budget': {null_budget_count}")


df['budget'].fillna(df['budget'].mean(), inplace=True)
df['gross'].fillna(df['gross'].mean(), inplace=True)


# # Remove rows with null values
# df = df.dropna()


# Check the number of null values for 'budget' column
null_budget_count = df['gross'].isnull().sum()
print(f"Number of null values for 'budget': {null_budget_count}")


# total_null_count = df.isnull().sum().sum() print(f"Total number of null values: {total_null_count}")



# Remove rows with null values
df = df.dropna()


# Write the cleaned data to another CSV file
df.to_csv('cleaned_movies_nneww.csv', index=False)