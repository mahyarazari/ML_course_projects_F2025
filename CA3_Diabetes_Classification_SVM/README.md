# Diabetes Prediction using Support Vector Machine (SVM)

## 1. Project Overview

This project implements a machine learning analysis to **predict diabetes** using the Pima Indians Diabetes dataset. The core steps include data preprocessing, scaling, dimensionality reduction using **PCA**, and training a **Support Vector Machine (SVM)** classification model with hyperparameter tuning.

## 2. Dataset

* **File Name**: `diabetes.csv` (This file must be available in the appropriate path)
* **Objective**: To predict the `Outcome` column, which indicates whether an individual has diabetes (1: Yes, 0: No).

## 3. Prerequisites

To run the notebook, you need a Python 3 environment and the following libraries. These are also listed in the `requirements.txt` file.

## 4. How to Run

All analysis and modeling are contained within the Jupyter Notebook **`ml_hw3_q8.ipynb`**.

1.  Clone the project repository.
2.  Install the necessary libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ensure the `diabetes.csv` file is accessible (according to the path specified in the notebook).
4.  Run the notebook **`ml_hw3_q8.ipynb`** in a Jupyter or Google Colab environment.

## 5. Key Implementation Details and Results

* **Data Preprocessing**: Zero values (treated as missing data) in features such as `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were imputed using the **mean** of the respective columns.
* **Feature Scaling**: **`MinMaxScaler`** was applied to scale all features.
* **Hyperparameter Tuning**: **`GridSearchCV`** was used for the SVM model with an RBF kernel, determining the following optimal hyperparameters:
    * **Best C**: 10
    * **Best $\gamma$ (Gamma)**: 0.1
* **Dimensionality Reduction (PCA)**: The data was reduced to **2 principal components** to allow for plotting the SVM decision boundary.
* **Model Results**: After optimization, the performance of the SVM model (with RBF kernel) was evaluated. Using `class_weight='balanced'` to address class imbalance, the model produced the following confusion matrix:
    ```
    [[121  29]
     [ 24  57]]
    ```
    * (121 correct predictions for Class 0, 57 correct predictions for Class 1)
