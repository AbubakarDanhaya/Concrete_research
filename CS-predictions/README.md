# Explainable Ensemble Learning for Concrete Compressive Strength Prediction Using Optimized Tree-Based Models

# Concrete Compressive Strength Prediction

## Overview
This project develops and evaluates machine learning models for predicting the compressive strength of concrete using its mix proportions and curing age. It also applies explainable AI techniques to understand how different input variables influence model predictions.

## Methodology

### 1. Data Collection and Preprocessing
The concrete dataset was loaded using Pandas. The following preprocessing steps were performed:

- Checked for missing values.
- Identified and removed duplicate records.
- Detected outliers using boxplots.
- Removed outliers using the Interquartile Range (IQR) method, with thresholds of 1.5 × IQR.
- Created an additional feature: water-to-cement ratio.

### 2. Exploratory Data Analysis (EDA)
Exploratory data analysis was conducted to understand the dataset and relationships between variables.

The analysis included:
- Descriptive statistical analysis.
- Feature distribution analysis using histograms.
- Correlation analysis.
- Correlation heatmap visualization.

### 3. Model Development
The dataset was divided into training (80%) and testing (20%) subsets using a fixed random state of 100.

Five individual machine learning models were developed:

1. AdaBoost Regressor
2. Random Forest Regressor
3. XGBoost Regressor
4. LightGBM Regressor
5. CatBoost Regressor

Hyperparameter optimization was performed using `RandomizedSearchCV` with 10-fold cross-validation.

Two ensemble learning approaches were also developed:

- **Voting Regressor:** Combines predictions from XGBoost, LightGBM, and CatBoost through averaging.
- **Stacking Regressor:** Combines Random Forest, XGBoost, and CatBoost predictions using RidgeCV as the final estimator.

### 4. Model Evaluation
Model performance was evaluated using the following metrics:

- **R²:** Measures the proportion of variance in concrete compressive strength explained by the model.
- **RMSE:** Measures the square root of the average squared prediction errors.
- **MAE:** Measures the average absolute prediction error.
- **Within ±20%:** Measures the percentage of predictions falling within ±20% of experimental values.

Model performance was further examined using:

- Experimental versus predicted strength plots.
- Residual distribution plots.

### 5. Model Explainability
Explainable AI techniques were applied to understand model behaviour and feature contributions.

The following methods were used:

- **SHAP (SHapley Additive exPlanations):** Evaluates the contribution and importance of input features to model predictions.
- **SHAP Summary Plot:** Shows the overall distribution and influence of feature contributions.
- **Partial Dependence Plots (PDP):** Examine the average relationship between individual input features and predicted compressive strength.

## Tools and Libraries

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- SciPy

## Workflow Summary

The overall research workflow follows these stages:

**Data Collection → Data Preprocessing → Exploratory Data Analysis → Feature Engineering → Model Development → Hyperparameter Optimization → Ensemble Learning → Model Evaluation → Explainability**

## Objective

The main objective is to develop reliable machine learning models for predicting concrete compressive strength and use explainable AI techniques to improve the interpretability of the predictions.