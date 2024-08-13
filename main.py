import os
import pandas as pd
import numpy as np
from models.Regression.XGBoost import best_model
from models.Regression.feature_scaling import preprocess_data
from colorama import init, Fore, Style


def begin_cli():
    init()
    os.system("cls" if os.name == "nt" else "clear")
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


def get_user_input():
    print(
        f"\n{Fore.YELLOW}Please enter the parameters to predict the revenue for your upcoming movie{Style.RESET_ALL}\n"
    )

    return {
        "released": input("When is the movie going to be released: "),
        "writer": input("Who is the writer of the movie: "),
        "rating": input("What is the MPAA rating of the movie: "),
        "name": input("What is the name of the movie: "),
        "genre": input("What is the genre of the movie: "),
        "director": input("Who is the director of the movie: "),
        "star": input("Who is the star of the movie: "),
        "country": input("Which country are you based: "),
        "company": input("Which company is producing the movie: "),
        "runtime": float(input("What is the runtime of the movie: ")),
        "score": float(
            input("How are the critics rating the movie in the initial screening: ")
        ),
        "budget": float(input("What is the budget of the movie: ")),
        "year": int(input("Which year is the movie shot: ")),
        "votes": float(
            input(
                "What is the total number of votes for the movie in the initial survey: "
            )
        ),
    }


def predict_gross(input_data):
    processed_data = preprocess_data(pd.DataFrame([input_data]))

    expected_features = best_model.feature_names_in_
    for feature in expected_features:
        if feature not in processed_data.columns:
            processed_data[feature] = 0  # or another appropriate default value

    processed_data = processed_data[expected_features]

    log_prediction = best_model.predict(processed_data)

    prediction = np.exp(log_prediction) - 1

    return prediction[0]


def predict_gross_range(gross):
    if gross <= 10000000:
        return f"Low Revenue (<= $10M)"
    elif gross <= 40000000:
        return f"Medium-Low Revenue ($10M - $40M)"
    elif gross <= 70000000:
        return f"Medium Revenue ($40M - $70M)"
    elif gross <= 120000000:
        return f"Medium-High Revenue ($70M - $120M)"
    elif gross <= 200000000:
        return f"Medium-High Revenue ($120M - $200M)"
    else:
        return f"Ultra High Revenue (>= $200M)"


if __name__ == "__main__":
    begin_cli()
    print(f"\n{Fore.YELLOW}Welcome Producer!!!{Style.RESET_ALL}\n")

    while True:
        input_data = get_user_input()
        predicted_gross = predict_gross(input_data)
        predicted_gross_range = predict_gross_range(predicted_gross)

        print(
            f'\n{Fore.GREEN}Predicted Revenue{Style.RESET_ALL} for "{Fore.CYAN}{input_data["name"]}{Style.RESET_ALL}": ${predicted_gross:,.2f}'
        )
        print(
            f"{Fore.GREEN}Predicted Revenue Range{Style.RESET_ALL}: {predicted_gross_range}\n"
        )

        another = input(
            f"{Fore.YELLOW}Would you like to predict another movie? (yes/no): {Style.RESET_ALL}"
        ).lower()
        if another != "yes":
            break

    print(
        f"\n{Fore.YELLOW}Thank you for using the Movie Revenue Prediction System!{Style.RESET_ALL}"
    )
