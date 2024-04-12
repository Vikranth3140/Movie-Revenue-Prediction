import pandas as pd

def merge_datasets_initial(a,b):
    b['movie_title'] = b['movie_title'].str.rstrip()
    merged_df = pd.merge(a, b, left_on='name', right_on='movie_title', how='inner')

    return merged_df

def merge_datasets_intermediate(a,b):
    merged_df = pd.merge(a, b, left_on='title', right_on='Movie_Title', how='inner')

    return merged_df

def merge_datasets_finalized(a,b):
    merged_df = pd.merge(a, b, left_on='Movie_Title', right_on='original_title', how='inner')

    return merged_df

def merge_datasets_finalized2(a,b):
    merged_df = pd.merge(a, b, left_on='movie_title', right_on='original_title', how='inner')    

    return merged_df

def remove_unnecessary_df1(df):
    cols_to_drop = [ 'company', 'runtime', 'rating','year','country','writer','votes','released']
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df

def remove_unnecessary_df2(df):
    cols_to_drop = [ 'plot_keywords', 'movie_imdb_link', 'language', "country",'content_rating', 'color', 'duration', 'facenumber_in_poster', 'movie_facebook_likes', 'num_voted_users', 'num_user_for_reviews', 'num_critic_for_reviews', 'title_year', 'aspect_ratio', 'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_facebook_likes', 'actor_1_facebook_likes', 'cast_total_facebook_likes',  'actor_3_name']
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df

def remove_unnecessary_df3(df):
    cols_to_drop = [ 'Censor', 'Runtime(Mins)', 'Rating','Year','side_genre']
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df

def remove_unnecessary_df4(df):
    cols_to_drop = [ 'rank', 'release_date','url','mpaa','theaters','runtime','year']
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df

def remove_unnecessary_df5(df):
    cols_to_drop = [ 'adult','belongs_to_collection','homepage','overview','popularity','poster_path','production_companies','production_countries','release_date','runtime','spoken_languages','status','tagline','title','video','vote_average','vote_count']
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df

def check_duplicates1(df1, df2):
    df2.rename(columns={'Movie_Title': 'name'}, inplace=True)

    df = pd.concat([df1, df2], axis=0)

    duplicated = df.duplicated(subset='name', keep=False)

    count = duplicated.sum()

    print(f'There are {count} duplicates between the two datasets.')
    df.drop_duplicates(subset='name', keep='first', inplace=True)

    return df

def check_duplicates2(df1, df2):
    df1.rename(columns={'title': 'name'}, inplace=True)  
    df2.rename(columns={'movie_title': 'name'}, inplace=True)
    df = pd.concat([df1, df2], axis=0)

    duplicated = df.duplicated(subset='name', keep=False)
    count = duplicated.sum()

    print(f'There are {count} duplicates between the two datasets.')
    df.drop_duplicates(subset='name', keep='first', inplace=True)

    return df

def check_duplicates3(df1, df2):
    df = pd.concat([df1, df2], axis=0)

    duplicated = df.duplicated(subset='name', keep=False)
    count = duplicated.sum()

    print(f'There are {count} duplicates between the two datasets.')
    df.drop_duplicates(subset='name', keep='first', inplace=True)

    return df

def write_to_csv(df, filename):
    df.to_csv(filename, index=False)

def count_budget_filled_rows(df):
    return df[pd.to_numeric(df['budget'], errors='coerce').notnull()].shape[0]

def main():
    df1 = pd.read_csv('SML-Project-1\datasets\Kaggle\movies.csv') ## Find where
    df2 = pd.read_csv('SML-Project-1\datasets\Kaggle\movie_metadata.csv')
    df3 = pd.read_csv('SML-Project-1\datasets\Kaggle\IMDb 5000+.csv')
    df4 = pd.read_csv('SML-Project-1\datasets\Kaggle\Top-500-movies.csv')
    df5 = pd.read_csv('SML-Project-1\datasets\Kaggle\movie_data_tmbd.csv')

    # 1) Initial Exploration and Cleaning
    cleaned_df1=remove_unnecessary_df1(df1)
    cleaned_df2=remove_unnecessary_df2(df2)
    num_rows_with_budget = count_budget_filled_rows(cleaned_df1)

    print(f"Number of rows with numeric 'budget': {num_rows_with_budget}")
    initial_dataset=merge_datasets_initial(cleaned_df1, cleaned_df2)
    # write_to_csv(initial_dataset, 'SML-Project-1\datasets\Initial\initial_dataset.csv')

    # 2) Thirst for More data
    cleaned_df3=remove_unnecessary_df3(df3)
    cleaned_df4=remove_unnecessary_df4(df4)
    intermediate_dataset=merge_datasets_intermediate(cleaned_df4, cleaned_df3)
    # write_to_csv(intermediate_dataset, 'SML-Project-1\datasets\Intermediate\intermediate_dataset.csv')
   
    cleaned_df5=remove_unnecessary_df5(df5)
    dataset=merge_datasets_finalized(cleaned_df3,cleaned_df5)
    # write_to_csv(dataset, 'SML-Project-1\datasets\dataset.csv')

    dataset2=merge_datasets_finalized2(cleaned_df2,cleaned_df5)
    # write_to_csv(dataset2, 'SML-Project-1\datasets\dataset2.csv')

    initial_merge=check_duplicates1(initial_dataset,dataset)
    intermediate_merge=check_duplicates2(intermediate_dataset,dataset2)
    Final_merge=check_duplicates3(initial_merge,intermediate_merge)

    # Further completing up Final_merge by manual scouring for budget or gross and using our 4 databases for more movies we arrive at final_dataset.

    # 3) Introducing final_dataset: Arrived by cleaning and dropping colums which were repetitive  and dropping null values

    # num_rows_with_budget = count_budget_filled_rows(cleaned_df)
    # print(f"Number of rows with numeric 'budget': {num_rows_with_budget}")
    # write_to_csv(Final_merge, 'SML-Project-1\datasets\Final_merge.csv')

if __name__ == "__main__":
    main()