# Heart Disease Prediction

## Project Overview

This project predicts whether a patient has heart disease using classical machine learning.

The dataset contains patient clinical features such as age, cholesterol, resting blood pressure, maximum heart rate, chest pain type, exercise-induced angina, and other medical indicators.

The original target variable had multiple disease severity levels. For this project, the task was reframed as binary classification:

- 0 = No heart disease
- 1 = Heart disease present

## Objective

The goal is to build, evaluate, and interpret a machine learning model for heart disease prediction.

This project focuses on learning the complete ML workflow:

- Loading and inspecting a real dataset
- Handling missing values
- Exploratory data analysis
- Feature and target separation
- Train/test splitting
- Model comparison
- Cross-validation
- Confusion matrix analysis
- Threshold tuning
- Saving trained model artifacts

## Dataset

The project uses the processed Cleveland heart disease dataset.

The dataset has:

- 303 patient records
- 13 input features
- 1 target column

Missing values were found in:

- ca: 4 missing values
- thal: 2 missing values

These missing values were filled using the mode because both columns are category-like coded variables.

## Features

Continuous features:

- age
- trestbps
- chol
- thalach
- oldpeak

Categorical / coded features:

- sex
- cp
- fbs
- restecg
- exang
- slope
- ca
- thal

Target:

- target

## Exploratory Data Analysis Findings

The dataset is reasonably balanced:

- No heart disease: 164 patients
- Heart disease: 139 patients

Chest pain type showed a strong association with the target. In particular, chest pain type 4 had the highest proportion of heart disease cases.

Maximum heart rate achieved (`thalach`) showed a negative relationship with heart disease. Patients with heart disease generally had lower maximum heart rate values in this dataset.

## Models Tested

The following models were compared:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest

Initial single train/test split results showed Random Forest performing best, but cross-validation gave a more reliable comparison.

## Cross-Validation Results

Using 5-fold cross-validation on the training set:

| Model | Mean CV Accuracy | Mean CV Precision | Mean CV Recall | Mean CV F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8304 | 0.8463 | 0.7743 | 0.8071 |
| KNN | 0.8098 | 0.8227 | 0.7561 | 0.7854 |
| Random Forest | 0.8055 | 0.8224 | 0.7379 | 0.7756 |
| Decision Tree | 0.7435 | 0.7350 | 0.6834 | 0.7078 |

Based on cross-validation, Logistic Regression was selected as the final model.

## Final Test Set Performance

Final model: Logistic Regression

Test accuracy:

```text
86.89%
```

Confusion matrix:

```text
[[27  6]
 [ 2 26]]
```

Meaning:

- True negatives: 27
- False positives: 6
- False negatives: 2
- True positives: 26

For heart disease screening, false negatives are especially important because they represent patients who actually have disease but were predicted as no disease.

## Threshold Tuning

The default decision threshold is 0.5.

Different thresholds were tested:

| Threshold | Accuracy | Precision | Recall | F1 | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.8197 | 0.7179 | 1.0000 | 0.8358 | 11 | 0 |
| 0.4 | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 6 | 2 |
| 0.5 | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 6 | 2 |
| 0.6 | 0.8852 | 0.8621 | 0.8929 | 0.8772 | 4 | 3 |
| 0.7 | 0.9016 | 0.8929 | 0.8929 | 0.8929 | 3 | 3 |

For a screening-focused use case, a lower threshold such as 0.3 may be preferred because it reduces false negatives, but it increases false positives.

## Saved Artifacts

The trained model and scaler were saved in the `models/` folder:

```text
models/logistic_regression_model.pkl
models/standard_scaler.pkl
```

Result files were saved in the `results/` folder:

```text
results/cross_validation_results.csv
results/threshold_results.csv
```

## Important Limitation

This project is educational and should not be used for real medical diagnosis.

The dataset is small and may not represent modern real-world clinical populations. A real clinical model would require a larger, validated, diverse dataset and medical approval.

## Skills Demonstrated

- Python
- Pandas
- Scikit-learn
- Data cleaning
- Exploratory data analysis
- Classification
- Model evaluation
- Cross-validation
- Threshold tuning
- Model saving with Joblib
