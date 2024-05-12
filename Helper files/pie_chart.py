import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('revised datasets\movies.csv')

country_counts = df['country'].value_counts()

fig, ax = plt.subplots(figsize=(10, 10))  # Make the plot bigger
ax.pie(country_counts, autopct='')  # Remove percentages

ax.legend(title="Countries", labels=country_counts.index[:10], loc='center left')

plt.show()
