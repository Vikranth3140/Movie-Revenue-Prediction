## Building a Custom Movie Dataset: A Journey from Raw Data to Insights

We have embarked on a data-driven odyssey to create a robust and tailored movie dataset. Let's delve into the steps we took to curate this treasure trove of cinematic information.

### 1. **Initial Exploration and Cleaning**

- **Starting Point**: We began with the "Movie Industry" dataset (retrieved from `movies.csv`). This dataset contained a whopping 7,669 movies.
- **Budget Focus**: Our first revelation was that approximately 5,300 movies had budget information available. We decided to focus on these films.
- **Null Values**: Rigorous inspection revealed null values across various input parameters. We meticulously cleaned and preprocessed the data, resulting in a refined set of around 3,200 movies.

### 2. **Budget Detective Work**

- **Cross-Dataset Comparison**: To augment our budget data, we compared our "Movie Industry" dataset with the "IMDb 5000 Movie Dataset" (`movie_metadata.csv`). By identifying similar movies, we inferred budgets for previously missing entries.
- **Top 500 Movies**: We scoured the budget of top 400-500 movies in "IMDb 5000+ Movies & Multiple Genres Dataset" (`IMDb 5000+.csv`) from the "Top 500 Movies by Production Budget" (`top-500-movies.csv`) to uncover an additional 400-500 movies with known budgets.

### 3. **Thirst for More Data**

- **Still Hungry**: Our hunger for data persisted. We needed more movies to enrich our dataset.
- **IMDb 5000**: We turned back to the "IMDb 5000 Movie Dataset" . Here, we found approximately 5,000 movies, many of which had budget information.
- **Manual Sleuthing**: We also found the budget of about 200 movies of "IMDb 5000+ Movies & Multiple Genres Dataset" by manually hunted down their budgets through diligent online searches.

### 4. **The Final Dataset Emerges**

- **8000 Movies**: Our relentless pursuit yielded a grand total of 8,000 movies.
- **Refinement**: Further data cleaning and preprocessing ensued. We meticulously organized the input parameters:
    - **Name**
    - **Genre**
    - **Director**
    - **Actor 1**
    - **Actor 2**
    - **IMDb Score**
    - **Budget**
- **Output Parameter**: The pièce de résistance was the inclusion of the **Revenue** as our output parameter.

### 5. **Introducing 'final_dataset'**

- **7117 Movies**: After the dust settled, our final dataset, aptly named 'final_dataset,' stood tall with 7,117 movies.
- **A Cinematic Goldmine**: Armed with this comprehensive dataset, we're now poised to unravel insights, predict box office success, and celebrate the magic of cinema.

---

Our journey from raw data to a refined dataset mirrors the art of filmmaking itself—meticulous, collaborative, and filled with discoveries. 🎬📊