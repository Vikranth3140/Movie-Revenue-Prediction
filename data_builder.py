import pandas as pd

def bulder():
    df1 = pd.read_csv('datasets\movies.csv')
    df2 = pd.read_csv('datasets\movie_metadata.csv')

    df2['movie_title'] = df2['movie_title'].str.rstrip()

    merged_df = pd.merge(df1, df2, left_on='name', right_on='movie_title', suffixes=('_file1', '_file2'), how='inner')

    merged_df.to_csv('result_dataset.csv', index=False)

def constructor():
    df = pd.read_csv('result_dataset.csv')
    df = df.drop(['movie_title', 'plot_keywords', 'movie_imdb_link', 'language', 'country_file1', 'country_file2', 'content_rating', 'color', 'duration', 'facenumber_in_poster', 'movie_facebook_likes', 'num_voted_users', 'num_user_for_reviews', 'num_critic_for_reviews', 'title_year', 'aspect_ratio', 'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_facebook_likes', 'actor_1_facebook_likes', 'cast_total_facebook_likes', 'company', 'released', 'votes', 'rating', 'writer', 'year', 'runtime', 'director_name', 'genres', 'imdb_score', 'star', 'actor_3_name'], axis=1)
    df.to_csv('final_dataset.csv', index=False)

bulder()
constructor()