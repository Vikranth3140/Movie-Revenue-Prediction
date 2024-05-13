# Movie Revenue Prediction

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

Official code for the movie revenue prediction project. Authors: [Vikranth Udandarao](mailto:vikranth22570@iiitd.ac.in) and [Pratyush Gupta](mailto:pratyush22375@iiitd.ac.in).

## Introduction

Accurately predicting a movie’s earnings is crucial for maximizing profitability in the contemporary film industry. This project aims to develop a machine learning model for predicting movie earnings based on input features like the movie name, MPAA rating, genre, year of release, IMDb rating, votes, director, writer, leading cast, country of production, budget, production company, and runtime. Using a structured methodology involving data collection, preprocessing, analysis, model selection, evaluation, and improvement, a robust predictive model is constructed. Various models, including Linear Regression, Decision Trees, Random Forest Regression, Bagging, XGBoosting, and Gradient Boosting, are trained and tested.

![](https://github.com/Vikranth3140/Movie-Revenue-Prediction/blob/main/figs/movie-revenue-prediction-diagram.png)

![Movie Revenue Prediction diagram](image.png)

## Getting Started

All our code was tested on Python 3.6.8 with Pytorch 1.9.0+cu111. Ideally, our scripts require access to a single GPU (uses `.cuda()` for inference). Inference can also be done on CPUs with minimal changes to the scripts.

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

Detailed instructions on how to set up our datasets are provided in [`data/DATA.md`](https://github.com/Vikranth3140/Movie-Revenue-Prediction/blob/main/data/DATA.md).

### Directory Structure

After setting up the datasets and the environment, the project root folder should look like this:
```
Movie-Revenue-Prediction/
|–– data
|–––– movies
|–––– ... other datasets
|–– features
|–– README.md
|–– data_preprocessing.py
|–– train_model.py
|–– evaluate_model.py
|–– main.py
|–– ... all other provided python scripts
```

## Running the Models

### Training the Models

You can train the models using:
```bash
python train_model.py --model <model_name> --dataset <dataset_path>
```
The `model_name` parameter can be one of [`linear_regression`, `decision_tree`, `random_forest`, `bagging`, `gradient_boost`, `xgboost`].

### Evaluating the Models

You can evaluate the trained models using:
```bash
python evaluate_model.py --model <model_name> --dataset <dataset_path>
```

## Data Preprocessing

We provide scripts for data preprocessing, including handling missing values, encoding categorical variables, and feature selection.

### Handling Missing Values

Missing values are handled using the `data_preprocessing.py` script:
```bash
python data_preprocessing.py --dataset <dataset_path> --output <output_path>
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
@inproceedings{udandarao2023movie-revenue,
  title={Movie Revenue Prediction},
  author={Udandarao, Vikranth and Gupta, Pratyush},
  booktitle={IIIT-Delhi},
  year={2023}
}
```

## Acknowledgements

The authors would like to extend their sincerest gratitude to Dr. A V Subramanyam (Computer Science & Engineering Dept., IIIT-Delhi) for their invaluable guidance throughout the project.

## Contact

Please feel free to open an issue or email us at [vikranth22570@iiitd.ac.in](mailto:vikranth22570@iiitd.ac.in) or [pratyush22375@iiitd.ac.in](mailto:pratyush22375@iiitd.ac.in).

---

Let me know if you need any additional visual elements such as pie charts or bar graphs, or if you have any other requests!