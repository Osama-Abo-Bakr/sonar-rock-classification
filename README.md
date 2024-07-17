# Sonar and Rock Classification

## Project Overview

This project aims to classify sonar returns from mines and rocks using machine learning techniques. The project includes data preprocessing, feature engineering, model training, and evaluation to achieve high classification accuracy.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Analysis and Preprocessing](#data-analysis-and-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Modeling](#modeling)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Results](#results)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Conclusion](#conclusion)

## Introduction

The goal of this project is to classify sonar returns from mines (0) and rocks (1) using machine learning models. The dataset consists of 208 samples, each with 60 features representing sonar returns.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Preprocessing, feature extraction, and model building
- **GridSearchCV**: Hyperparameter tuning

## Data Analysis and Preprocessing

1. **Data Loading**:
   - Loaded the dataset using `pd.read_csv()`.

2. **Data Summary**:
   - Checked for missing values using `data.isnull().sum()`.
   - Summarized the data using `data.info()` and `data.describe()`.

3. **Label Encoding**:
   - Encoded the target labels (Rock, Mine) using `LabelEncoder()`.

4. **Data Visualization**:
   - Visualized data correlations using heatmaps and box plots.

## Feature Engineering

1. **Label Encoding**:
   - Encoded categorical labels (Rock, Mine) into numerical labels (1, 0) using `LabelEncoder()`.

2. **Data Splitting**:
   - Split the data into training and testing sets using `train_test_split()`.

## Modeling

1. **Logistic Regression**:
   - Built a Logistic Regression model using `LogisticRegression()`.
   - Evaluated the model performance on training and testing sets.

2. **Random Forest Classifier**:
   - Built a Random Forest model using `RandomForestClassifier()`.
   - Evaluated the model performance on training and testing sets.

## Hyperparameter Tuning

1. **GridSearchCV for Logistic Regression**:
   - Tuned hyperparameters for the Logistic Regression model using `GridSearchCV()`.
   - ```
     GridSearchCV(cv=5, estimator=LogisticRegression(), n_jobs=-1,
             param_grid={'C': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2],
                         'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag',
                                    'saga']},
             verbose=6)
     ```

2. **GridSearchCV for Random Forest**:
   - Tuned hyperparameters for the Random Forest model using `GridSearchCV()`.

## Results

- **Model Accuracy**:
  - Achieved high accuracy with both Logistic Regression and Random Forest models.
  - The best hyperparameters for each model were identified using GridSearchCV.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/sonar-rock-classification.git
   ```

2. Navigate to the project directory:
   ```bash
   cd sonar-rock-classification
   ```


## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Models**:
   - Run the provided script to train the models and evaluate their performance.

3. **Predict Outcomes**:
   - Use the trained models to classify new sonar return samples.

## Conclusion

This project demonstrates the application of machine learning techniques for classifying sonar returns from mines and rocks. By leveraging data preprocessing, feature engineering, and hyperparameter tuning, high accuracy was achieved in classification.

---

### Sample Code (for reference)

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Reading Data
data = pd.read_csv("D:\\Courses language programming\\Machine Learning\\Folder Machine Learning\\Sonar&Mine\\Copy of sonar data.csv", header=None)
data.head(10)

# Data Preprocessing
data[60].value_counts()
data[60] = LabelEncoder().fit_transform(data[60])
x_input = data.drop(60, axis=1)
y_output = data[60]
x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, train_size=0.7, random_state=42)

# Logistic Regression
model_logi = LogisticRegression()
model_logi.fit(x_train, y_train)
print(model_logi.score(x_train, y_train))
print(model_logi.score(x_test, y_test))

# Hyperparameter Tuning for Logistic Regression
param = {"penalty": ['l1', 'l2', 'elasticnet', "none"], "C": [0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 0.9, 1, 1.2], "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
new_model_log = GridSearchCV(estimator=model_logi, param_grid=param, verbose=6, cv=5, n_jobs=-1)
new_model_log.fit(x_train, y_train)
print(new_model_log.best_estimator_, new_model_log.best_score_)

# Random Forest Classifier
model2_RF = RandomForestClassifier()
model2_RF.fit(x_train, y_train)
print(model2_RF.score(x_train, y_train))
print(model2_RF.score(x_test, y_test))

# Hyperparameter Tuning for Random Forest
param2 = {"n_estimators": np.arange(22, 27, 1), "max_depth": np.arange(11, 15, 1), "min_samples_split": np.arange(2,4), "min_samples_leaf": np.arange(2,4)}
new_model_RF = GridSearchCV(estimator=model2_RF, param_grid=param2, verbose=6, cv=5, n_jobs=-1)
new_model_RF.fit(x_train, y_train)
print(new_model_RF.best_estimator_, new_model_RF.best_score_)
```

---
