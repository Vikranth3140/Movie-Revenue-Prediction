import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("masti.csv")

columns_to_drop = [
    "budget_x",
    "gross_x",
    "director_name",
    "genres",
    "movie_title",
    "star",
    "imdb_score",
]
df = df.drop(columns_to_drop, axis=1)

df = df.drop_duplicates()

df = df.dropna()

df = df.rename(columns={"budget_y": "budget", "gross_y": "gross"})

df.to_csv("updated_masti.csv", index=False)


df = pd.read_csv("new_updated_less-than-350m-dataset.csv")

plt.figure(figsize=(10, 6))
plt.scatter(df["budget"], df["gross"], color="blue", alpha=0.5)
plt.title("Gross vs Budget")
plt.xlabel("Budget")
plt.ylabel("Gross")
plt.grid(True)
plt.show()


df = pd.read_csv("oldd_dataset.csv")

df = df[df["budget"] <= 1000000000]

df.to_csv("new_updated_less-than-1b-dataset.csv", index=False)


df = pd.read_csv("oldd_dataset.csv")

df = df[df["budget"] <= 350000000]

df.to_csv("new_updated_less-than-350m-dataset.csv", index=False)
