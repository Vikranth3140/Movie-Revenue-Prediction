## Building a Custom Movie Dataset: A Journey from Raw Data to Insights

We have embarked on a data-driven odyssey to create a robust and tailored movie dataset. Let's delve into the steps we took to curate this treasure trove of cinematic information.

### 1. **Initial Exploration and Cleaning**

- **Starting Point**: We began with the "Movie Industry" dataset (retrieved from `movies.csv`). This dataset contained a whopping 7,669 movies.
- **Budget Focus**: Our first revelation was that approximately 5,300 movies had budget information available. We decided to focus on these films.
- **Needing More Accurate Data**: Even though we had 5300 movies with budget known, we needed more than one actor who acted in the movie, so we referenced "IMDb 5000 Movie Dataset" (`movie_metadata.csv`) as it had second actor name also and found 3588 movies which were similair between the 2 datasets. By identifying similar movies, we had 3588 movies which were complete by our standards hence becoming our initial dataset.

### 2. **Thirst for More Data**

- **Still Hungry**: Our hunger for data persisted. We needed more movies to enrich our dataset.
- **Top 500 Movies**: We scoured the budget of top 400-500 movies in "IMDb 5000+ Movies & Multiple Genres Dataset" (`IMDb 5000+.csv`) from the "Top 500 Movies by Production Budget" (`Top-500-movies.csv`) to uncover an additional 400-500 movies.
- **The Movies Dataset**: We turned back to the "Movies Dataset" (`movies.csv`) and comparing it with "IMDb 5000+ Movies & Multiple Genres Dataset" to find more movies and found about 2500 movies without duplicates . So in total we had about 6200 movies.
- **Manual Sleuthing**: Further finalizing incomplete data(not having gross or budget) by manual scouring the web and using our datasets we reached our final dataset.

### 3. **The Final Dataset Emerges: Introducing 'final_dataset'**

- **7119 Movies**: Our relentless pursuit yielded a grand total of 7119 movies.
- **Refinement**: Further data cleaning and preprocessing ensued. We meticulously organized the input parameters:
  - **Name**
  - **Genre**
  - **Director**
  - **Actor 1**
  - **Actor 2**
  - **IMDb Score**
  - **Budget**
- **Output Parameter**: The pièce de résistance was the inclusion of the **Revenue** as our output parameter.
- **7119 Movies**: After the dust settled, our final dataset, aptly named 'final_dataset,' stood tall with 7,119 movies.
- **A Cinematic Goldmine**: Armed with this comprehensive dataset, we're now poised to unravel insights, predict box office success, and celebrate the magic of cinema.

---

Our journey from raw data to a refined dataset mirrors the art of filmmaking itself—meticulous, collaborative, and filled with discoveries. 🎬📊
