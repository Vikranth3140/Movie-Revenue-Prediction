import pandas as pd

def merge_datasets():
    df1 = pd.read_csv('movies.csv')
    df2 = pd.read_csv('movie_metadata.csv')

    df2['movie_title'] = df2['movie_title'].str.rstrip()

    merged_df = pd.merge(df1, df2, left_on='name', right_on='movie_title', how='inner')

    return merged_df

def remove_duplicates(df):
    cols_to_drop = [ 'plot_keywords', 'movie_imdb_link', 'language', "country",'content_rating', 'color', 'duration', 'facenumber_in_poster', 'movie_facebook_likes', 'num_voted_users', 'num_user_for_reviews', 'num_critic_for_reviews', 'title_year', 'aspect_ratio', 'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_facebook_likes', 'actor_1_facebook_likes', 'cast_total_facebook_likes',  'actor_3_name']
    df.drop(cols_to_drop, axis=1, inplace=True)


    return df
def remove_col(df):
    cols_to_drop = [ 'Censor', 'Runtime(Mins)', 'Rating','Year','side_genre']
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df

def add_gross_column(df):
    if 'gross_x' in df.columns:
        df['gross'] = df['gross_x']
    elif 'gross_y' in df.columns:
        df['gross'] = df['gross_y']
    return df

def add_budget_column(df):
    if 'budget_y' in df.columns:
        df['budget'] = df['budget_y']
    elif 'gross_x' in df.columns:
        df['budget'] = df['budget_x']
    return df

def remove_incomplete_rows(df):
    cols_to_check = ['name', 'genre', 'score', 'director', 'actor_2_name', 'actor_1_name', 'gross', 'budget']
    df.dropna(subset=cols_to_check, how='any', inplace=True)
    return df

def remove_columns(df, cols):
    existing_cols = [col for col in cols if col in df.columns]
    df.drop(existing_cols, axis=1, inplace=True)
    return df

def write_to_csv(df, filename):
    df.to_csv(filename, index=False)
def count_budget_filled_rows(df):
    return df[pd.to_numeric(df['budget'], errors='coerce').notnull()].shape[0]

def main():
    df1 = pd.read_csv('SML-Project-1\datasets\IMDb 5000+.csv')
    #merged_df = merge_datasets()
    cleaned_df = remove_col(df1)
    # compared_df = add_gross_column(cleaned_df)
    # dropped_df = add_budget_column(compared_df)
    # columns_to_remove = ['budget_x', 'gross_x', 'budget_y', 'gross_y']
    # dropped_df = remove_columns(dropped_df, columns_to_remove)
    # final_df = remove_incomplete_rows(dropped_df)
    # num_rows_with_budget = count_budget_filled_rows(cleaned_df)
    # print(f"Number of rows with numeric 'budget': {num_rows_with_budget}")
    write_to_csv(cleaned_df, 'SML-Project-1\datasets\IMDb 5000+.csv')

if __name__ == "__main__":
    main()