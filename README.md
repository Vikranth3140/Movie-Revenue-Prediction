# Movie Revenue Prediction

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![sk-learn](https://img.shields.io/badge/scikit-learn-grey.svg?logo=scikit-learn)](https://scikit-learn.org/stable/whats_new.html)


## Introduction

Accurately predicting a movie’s earnings is crucial for maximizing profitability in the contemporary film industry. This project aims to develop a machine learning model for predicting movie earnings based on input features like the movie name, MPAA rating, genre, year of release, IMDb rating, votes, director, writer, leading cast, country of production, budget, production company, and runtime. Using a structured methodology involving data collection, preprocessing, analysis, model selection, evaluation, and improvement, a robust predictive model is constructed. Various models, including Linear Regression, Decision Trees, Random Forest Regression, Bagging, XGBoosting, and Gradient Boosting, are trained and tested.

![Movie Revenue Prediction diagram](fig/intro.png)


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
Detailed instructions on how to set up our datasets are provided in [old datasets\README.md](old%20datasets/README.md).


### Directory Structure

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
│   ├── gradient_boost.py
│   ├── linear_regression_pca.py
│   ├── linear_regression.py
│   ├── random_forest.py
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
│   └── output.csv
│
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```


## Running the Models

You can run the models using:

```bash
python <model_name>.py
```

The `model_name` parameter can be one of [`linear_regression`, `decision_tree`, `random_forest`, `decision_tree_bagging`, `gradient_boost`, `XGBoost`].

## Data Preprocessing

We provide scripts for data preprocessing, including handling missing values, encoding categorical variables, and feature selection.

### Handling Missing Values

Missing values are handled using the `data_preprocessing.py` script:
```bash
python data_preprocessing.py
```

### Encoding Categorical Variables

Categorical variables are encoded using Label Encoding. This is implemented in the `data_preprocessing.py` script.

### Feature Selection

We use SelectKBest for feature selection, as implemented in the `data_preprocessing.py` script.

## Model Improvement

We employ several strategies for model improvement, including standardizing data, applying logarithmic transformations, and hyperparameter tuning using GridSearchCV.

### Standardizing Data

To ensure consistent scaling across features, we use Standard Scaler.

### Logarithmic Transformations

Logarithmic transformations are applied to skewed data (e.g., budget and gross revenue).

### Hyperparameter Tuning

Hyperparameter tuning is performed using GridSearchCV to optimize model parameters.

## Command Line Interface (CLI)

A CLI is developed to allow users to input movie features and get revenue predictions. Users can select different models for prediction.

### Running the CLI

1. Navigate to the project directory.
2. Run the CLI:
   ```bash
   python main.py
   ```
3. Follow the prompts to input the movie features and choose the prediction model.

## Results

The Gradient Boosting model achieved the best performance with:
- **Training Accuracy:** 91.58%
- **Testing Accuracy:** 82.42%

The model evaluation results for all models are as follows:

| Model           | Training R² | Training MAPE | Testing R² | Testing MAPE |
|-----------------|-------------|---------------|------------|--------------|
| Linear Regression | 0.6553      | 35.23%        | 0.6706     | 18.49%       |
| Decision Tree     | 0.8664      | 13.00%        | 0.6947     | 4.60%        |
| Bagging           | 0.8583      | 13.32%        | 0.7719     | 5.67%        |
| Gradient Boosting | 0.9158      | 10.57%        | 0.8242     | 5.69%        |
| XGBoosting        | 0.9079      | 9.70%         | 0.8102     | 5.53%        |
| Random Forest     | 0.8728      | 14.29%        | 0.7786     | 5.33%        |

## Conclusion

The developed Gradient Boosting model demonstrates promising accuracy and generalization capabilities, facilitating informed decision-making in the film industry to maximize profits.

## Citation

If you found this work useful, please consider citing it as:
```
@misc{udandarao2024movie-revenue,
  title={Movie Revenue Prediction},
  author={Udandarao, Vikranth and Gupta, Pratyush},
  year={2024},
  howpublished={\url{https://github.com/Vikranth3140/Movie-Revenue-Prediction}}
}
```


## Contact

Please feel free to open an issue or email me at [vikranth22570@iiitd.ac.in](mailto:vikranth22570@iiitd.ac.in).


## License

This project is licensed under the [MIT License](LICENSE).
