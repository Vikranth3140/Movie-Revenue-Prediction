import os
import pandas as pd
import numpy as np
from models.gradient_boost import best_model, le
from colorama import init, Fore, Style

def begin_cli():
    init()

    os.system('cls' if os.name == 'nt' else 'clear')

    title = (
        f"{Fore.CYAN}{Style.BRIGHT}\n"
        " __  __            _        ____                                                   \n"
        "|  \/  | _____   _(_) ___  |  _ \ _____   _____ _ __  _   _  ___                    \n"
        "| |\/| |/ _ \ \ / / |/ _ \ | |_) / _ \ \ / / _ \ '_ \| | | |/ _ \                   \n"
        "| |  | | (_) \ V /| |  __/ |  _ <  __/\ V /  __/ | | | |_| |  __/                   \n"
        "|_|__|_|\___/ \_/ |_|\___| |_| \_\___| \_/ \___|_|_|_|\__,_|\___|_                  \n"
        "|  _ \ _ __ ___  __| (_) ___| |_(_) ___  _ __   / ___| _   _ ___| |_ ___ _ __ ___  \n"
        "| |_) | '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \  \___ \| | | / __| __/ _ \ '_ ` _ \ \n"
        "|  __/| | |  __/ (_| | | (__| |_| | (_) | | | |  ___) | |_| \__ \ ||  __/ | | | | |\n"
        "|_|   |_|  \___|\__,_|_|\___|\__|_|\___/|_| |_| |____/ \__, |___/\__\___|_| |_| |_|\n"
        "                                                      |___/                       \n"
        f"{Style.RESET_ALL}"
    )
    print(title)

# Define the function to preprocess input data
def preprocess_input(released, writer, rating, name, genre, director, star, country, company, runtime, score, budget, year, votes):
    # Transform categorical features using LabelEncoder
    released_encoded = le.fit_transform([released])
    writer_encoded = le.fit_transform([writer])
    rating_encoded = le.fit_transform([rating])
    name_encoded = le.fit_transform([name])
    genre_encoded = le.fit_transform([genre])
    director_encoded = le.fit_transform([director])
    star_encoded = le.fit_transform([star])
    country_encoded = le.fit_transform([country])
    company_encoded = le.fit_transform([company])

    # Create a DataFrame with the preprocessed input
    input_data = pd.DataFrame({
        'released': released_encoded,
        'writer': writer_encoded,
        'rating': rating_encoded,
        'name': name_encoded,
        'genre': genre_encoded,
        'director': director_encoded,
        'star': star_encoded,
        'country': country_encoded,
        'company': company_encoded,
        'runtime': runtime,
        'score': score,
        'budget': budget,
        'year': year,
        'votes': votes,
    })

    return input_data

# Function to predict the gross range
def predict_gross_range(input_data):
    predicted_gross = best_model.predict(input_data)
    if predicted_gross <= 5000000:
        return f"Low Revenue (\u2264 $5M)"
    elif predicted_gross <= 25000000:
        return f"Medium-Low Revenue ($5M - $25M)"
    elif predicted_gross <= 50000000:
        return f"Medium Revenue ($25M - $50M)"
    elif predicted_gross <= 80000000:
        return f"Medium Revenue ($50M - $80M)"
    else:
        return f"High Revenue ($25M - $50M)"

# Example usage
if __name__ == "__main__":
    # Initialize the CLI
    begin_cli()

    # Take user input
    print(f'\n{Fore.YELLOW}Welcome Producer!!!{Style.RESET_ALL}\n')
    print(f'\n{Fore.YELLOW}Please enter the parameters to predict the revenue range for your upcoming movie{Style.RESET_ALL}\n')
    
    released = input("When is the movie going to be released: ")
    writer = input("Who is the writer of the movie: ")
    rating = input("What is the MPAA rating of the movie: ")
    name = input("What is the name of the movie: ")
    genre = input("What is the genre of the movie: ")
    director = input("Who is the director of the movie: ")
    star = input("Who is the star of the movie: ")
    country = input("Which country are you based: ")
    company = input("Which company is producing the movie: ")
    runtime = float(input("What is the runtime of the movie: "))
    score = float(input("How are the critics rating the movie in the initial screening: "))
    budget = float(input("What is the budget of the movie: "))
    year = int(input("Which year is the movie shot: "))
    votes = float(input("What is the total number of votes for the movie in the initial survey: "))

    # Preprocess the input data
    input_data = preprocess_input(released, writer, rating, name, genre, director, star, country, company, runtime, score, budget, year, votes)

    # Predict the revenue range
    predicted_gross_range = predict_gross_range(input_data)
    print(f'\n{Fore.GREEN}Predicted Revenue Range{Style.RESET_ALL} for "{Fore.CYAN}{name}{Style.RESET_ALL}": {predicted_gross_range}\n')