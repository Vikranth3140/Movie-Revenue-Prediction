def calculate_gross(name, genre, director, star, country, company, runtime, score, budget, year, votes):
    # Calculate gross using a simple formula (this is just an example)
    gross = budget + (score * votes) / 1000
    return gross

def main():
    # Get user input for movie parameters
    name = input("Enter movie name: ")
    genre = input("Enter movie genre: ")
    director = input("Enter director's name: ")
    star = input("Enter star's name: ")
    country = input("Enter country: ")
    company = input("Enter production company: ")
    runtime = int(input("Enter runtime in minutes: "))
    score = float(input("Enter IMDb score: "))
    budget = float(input("Enter budget in millions: "))
    year = int(input("Enter release year: "))
    votes = int(input("Enter number of votes: "))

    # Calculate gross
    gross = calculate_gross(name, genre, director, star, country, company, runtime, score, budget, year, votes)
    print(f"The gross for {name} is ${gross} million.")

if __name__ == "__main__":
    main()