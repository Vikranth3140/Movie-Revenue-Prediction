# Movie Revenue Prediction

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![sk-learn](https://img.shields.io/badge/scikit-learn-grey.svg?logo=scikit-learn)](https://scikit-learn.org/stable/whats_new.html)
[![arXiv](https://img.shields.io/badge/arXiv-2405.11651-b31b1b.svg)](https://arxiv.org/abs/2405.11651)
[![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-brightgreen.svg)](https://movie-revenue-prediction.streamlit.app)

[Academia Paper](https://www.academia.edu/119091410/Movie_Revenue_Prediction)

## Introduction

In the contemporary film industry, accurately predicting a movie's earnings is paramount for maximizing profitability. This project aims to develop a sophisticated machine learning model to forecast movie earnings based on a comprehensive set of input features, including the movie name, MPAA rating, genre, year of release, IMDb rating, votes by the watchers, director, writer, leading cast, country of production, budget, production company, and runtime.

Numerous factors influence a movie's earnings, and the optimal combination of these factors remains elusive. Our machine learning model seeks to uncover the most significant factors for box office success by analyzing real data from a diverse array of movies produced globally.

We hypothesize that certain parameters, such as the director's track record and the genre of the film, hold more significance in predicting movie revenue than others. Observations suggest that despite lower IMDb ratings, action-oriented films often perform well at the box office, while genres such as comedy or emotional dramas, despite potentially higher IMDb ratings, may not achieve comparable revenue outcomes. These insights highlight the complex interplay between film attributes and audience preferences.

By leveraging these diverse attributes, our goal is to construct a robust predictive model that can offer valuable insights and aid decision-making in the film industry. Ultimately, this will help filmmakers optimize their movie production strategies for maximum profit and popularity.

<img src="fig/intro.png" alt="Movie Revenue Prediction diagram" width="500" height="400">

## Directory Structure

```
Movie-Revenue-Prediction/
│
├── fig
│   └─ intro.png
│
├── Helper Files
│   ├── Best Features
│   │   ├── feature_scores.py
│   │   ├── feature_scores.txt
│   │   ├── significant_features.py
│   │   └── significant_features.txt
│   ├── budgetxgross.py
│   ├── data_visualization.py
│   ├── gross_histogram.py
│   ├── null_values_check.py
│   └── pie_chart.py
│
├── Misc
│   └─ initial_try.py
│
├── models
│   ├── accuracies.txt
│   ├── decision_tree_bagging.py
│   ├── decision_tree.py
│   ├── feature_scaling.py
│   ├── gradient_boost.py
│   ├── linear_regression_pca.py
│   ├── linear_regression.py
|   ├── random_forest.py
│   ├── tracking_XGBoost.py
│   └── XGBoost.py
│
├── old datasets
│   ├── finalised dataset
│   │   ├── dataset_modified.py
│   │   ├── masti.csv
│   │   ├── new_updated_less-than-1b-dataset.csv
│   │   ├── new_updated_less-than-350m-dataset.csv
│   │   ├── old_data.csv
│   │   └── updated_masti.csv
│   ├── initial
│   │   ├── initial_dataset.csv
│   │   └── initial_merge.csv
│   ├── Intermediate
│   │   ├── intermediate_dataset.csv
│   │   └── intermediate_merge.csv
│   │   └── intermediate1_dataset.csv
│   ├── Kaggle
│   │   ├── IMDb 5000+.csv
│   │   ├── movie_data_imdb.csv
│   │   ├── movie_metadata.csv
│   │   └── top_500_movies.csv
│   │
│   ├── data_builder_check.py
│   ├── dataset.csv
│   ├── dataset2.csv
│   ├── final_dataset.csv
│   ├── final_merge.csv
│   └── README.md
│
├── Reports
│   ├── Proposal
│   │   ├── proposal.pdf
│   │   └── proposal.tex
│   ├── 1st Project Report
│   │   ├── 1st_Project_Report.pdf
│   │   ├── 1st_Project_Report.tex
│   │   └── pics
│   │       ├── gross_histogram.png
│   │       ├── k_best.png
│   │       ├── model_accuracy_plot.png
│   │       ├── null_values.png
│   └── Final Report
│       ├── Final_Report.pdf
│       ├── Final_Report.tex
│       └── pics
│           ├── gross_histogram.png
│           ├── k_best.png
│           ├── model_accuracy_plot.png
│           ├── null_values.png
│           ├── pie_chart.png
│           └── R2_score_tracking.png
│
├── revised datasets
│   ├── movies.csv
|   ├── output.csv
│   └── README.md
│
├── .gitignore
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
└── streamlit_app.py
```

## Getting Started

All our code was tested on Python 3.6.8 with scikit-learn 1.3.2. Ideally, our scripts require access to a single GPU (uses `.cuda()` for inference). Inference can also be done on CPUs with minimal changes to the scripts.

### Setting up the Environment

We recommend setting up a Python virtual environment and installing all the requirements. Please follow these steps to set up the project folder correctly:

```bash
git clone https://github.com/Vikranth3140/Movie-Revenue-Prediction.git
cd Movie-Revenue-Prediction

python3 -m venv ./env
source env/bin/activate

pip install -r requirements.txt
```

### Setting up Datasets

The datasets have been taken from [Movie Industry](https://www.kaggle.com/datasets/danielgrijalvas/movies) dataset.
\
Detailed instructions on how to set up our revised datasets are provided in [revised datasets\README.md](revised%20datasets/README.md).

## Running the Models

You can run the models using:

```bash
python <model_name>.py
```

The `model_name` parameter can be one of [`linear_regression`, `decision_tree`, `random_forest`, `decision_tree_bagging`, `gradient_boost`, `XGBoost`].

## Data Preprocessing

We provide scripts for data preprocessing, including handling missing values, encoding categorical variables, feature scaling, and feature engineering.

### Handling Missing Values

Missing values are handled using the `SimpleImputer` with a median strategy in the `feature_scaling.py` script:

### Encoding Categorical Variables

Categorical variables are encoded using Label Encoding. This is implemented in the `feature_scaling.py` script which is called before the training of every model

### Feature Scaling

Enhanced feature preprocessing is implemented in `feature_scaling.py`:

Log transformation for skewed numerical features, particularly budget and revenue
StandardScaler applied to normalize numerical features

These preprocessing steps have resulted in substantially improved model performance, with significantly lower Mean Absolute Percantage Error(MAPE) and Mean Squared Logarithmic Error (MSLE).

### Feature Engineering

New features are created in our models:

- vote_score_ratio
- budget_year_ratio
- vote_year_ratio
- score_runtime_ratio
- budget_per_minute
- votes_per_year

Binary features introduced:

- is_recent
- is_high_budget
- is_high_votes
- is_high_score

These engineered features capture complex relationships and trends in the data, enhancing our model's ability to discover patterns. The combination of ratio-based and binary features provides a richer representation of the movie attributes, leading to improved predictive performance across our various models

### Feature Selection

We use SelectKBest for helping us know which features contribute the most towards our target variable, as implemented in the `significant_features.py` and `feature_scores.py` scripts.

## Model Improvement

We employed strategies such as hyperparameter tuning using GridSearchCV for model improvement.

### Hyperparameter Tuning

Hyperparameter tuning is performed using GridSearchCV to optimize model parameters.

## Command Line Interface (CLI)

We have developed a Command Line Interface (CLI) to allow users to input movie features and get revenue predictions. This tool provides an estimate of the inputted movie's revenue within specific ranges:

- Low Revenue: <= $10M
- Medium-Low Revenue: $10M - $40M
- Medium Revenue: $40M - $70M
- Medium-High Revenue: $70M - $120M
- High Revenue: $120M - $200M
- Ultra High Revenue: >= $200M

### Using the CLI

1. Navigate to the project directory.
2. Run the CLI:
   ```bash
   python main.py
   ```
3. Follow the prompts to input the movie features and choose the prediction model.

## Streamlit Web Interface

Additionally a web interface is also developed using Streamlit to allow users to input movie features and get revenue predictions.

### Running the Web Interface

1. Navigate to the project directory.
2. Run the Web Interface:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Follow the prompts to input the movie features and choose the prediction model.

## Model Evaluation Results

We evaluated our models using two key metrics: R² Score (Coefficient of Determination) and MSLE (Mean Squared Logarithmic Error). Here are the results for each model:

| Model             | Training R² | Training MSLE | Testing R² | Testing MSLE |
| ----------------- | ----------- | ------------- | ---------- | ------------ |
| Linear Regression | 0.6181      | 0.0053        | 0.6520     | 0.0051       |
| Decision Tree     | 0.8310      | 0.0024        | 0.5994     | 0.0059       |
| Bagging           | 0.8380      | 0.0023        | 0.7105     | 0.0042       |
| Gradient Boosting | 0.8750      | 0.0016        | 0.7350     | 0.0040       |
| XGBoosting        | 0.8633      | 0.0018        | 0.7402     | 0.0041       |
| Random Forest     | 0.8475      | 0.0022        | 0.7235     | 0.0041       |

## Conclusion

The developed Gradient Boosting and XGBoost models demonstrates promising accuracy and generalization capabilities, facilitating informed decision-making in the film industry to maximize profits.

## Citation

If you found this work useful, please consider citing it as:

```
@misc{udandarao2024movie,
      title={Movie Revenue Prediction using Machine Learning Models},
      author={Vikranth Udandarao and Pratyush Gupta},
      year={2024},
      eprint={2405.11651},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact

Please feel free to open an issue or email us at [vikranth22570@iiitd.ac.in](mailto:vikranth22570@iiitd.ac.in) or [pratyush22375@iiitd.ac.in](mailto:pratyush22375@iiitd.ac.in).

## License

This project is licensed under the [MIT License](LICENSE).
