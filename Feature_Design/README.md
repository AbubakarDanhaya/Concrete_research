# 🧪 Improving Machine Learning Prediction of Concrete Compressive Strength Using Engineering-Based Feature Design

## 📌 Overview

This repository presents a machine learning framework for predicting the compressive strength of concrete by integrating **engineering-informed feature design** with data-driven modeling.

Unlike conventional approaches that rely solely on raw input variables, this study introduces **physically meaningful engineered features**—such as water–cement ratio and binder content—to enhance both predictive performance and interpretability.

The results demonstrate that incorporating engineering knowledge into feature design significantly improves model accuracy and aligns predictions with established concrete mechanics principles.

---

## 🎯 Objectives

* Develop robust machine learning models for predicting concrete compressive strength
* Introduce **engineering-based features** to improve model performance
* Compare raw and engineered feature sets systematically
* Provide **interpretable insights** using SHAP (Shapley Additive Explanations)
* Bridge the gap between **data-driven modeling and engineering understanding**

---

## 📂 Dataset

* Source: UCI Machine Learning Repository
* Dataset: Concrete Compressive Strength
* Samples: 1030
* Input Variables: 8
* Output Variable: Compressive Strength (MPa)

### Input Features

* Cement
* Blast Furnace Slag
* Fly Ash
* Water
* Superplasticizer
* Coarse Aggregate
* Fine Aggregate
* Age

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Data inspection and cleaning
* Duplicate removal
* Outlier detection using IQR
* Optional dataset variation (with/without outliers)
* Feature scaling (for linear models)

---

### 2. Engineering-Based Feature Design (Core Contribution)

The following features were derived based on concrete mix design principles:

* **Water–Cement Ratio (w/c)**
* **Binder Content** = Cement + Slag + Fly Ash
* **Water–Binder Ratio (w/b)**
* **Aggregate Ratio**

Two datasets were created:

* Raw feature set
* Enhanced feature set (raw + engineered features)

---

### 3. Model Development

The following models were implemented:

* Linear Regression (baseline)
* Random Forest Regressor
* XGBoost Regressor

---

### 4. Model Evaluation

Models were evaluated using:

* R² (Coefficient of Determination)
* RMSE (Root Mean Square Error)
* MAE (Mean Absolute Error)

Additionally, **predicted vs experimental plots** with ±20% deviation bounds were used for visual assessment.

---

### 5. Model Interpretability

SHAP (SHapley Additive exPlanations) was used to:

* Identify key influencing features
* Analyze feature impact direction
* Validate engineering consistency of predictions

---

## 📊 Key Results

* Engineered features significantly improved model performance

* Models trained on enhanced features showed:

  * Higher R² values
  * Lower RMSE and MAE
  * Reduced prediction scatter

* SHAP analysis confirmed:

  * Water–cement ratio as the most influential feature
  * Binder content positively contributes to strength
  * Strong alignment with concrete mechanics theory

---

## 📈 Visualization

The repository includes:

* Predicted vs Experimental plots (Raw vs Engineered)
* SHAP Summary plots
* SHAP Feature Importance (bar plots)
* SHAP Dependence plots
* Waterfall plots (local interpretability)

---

## 🧠 Key Insight

This work demonstrates that:

> Incorporating **engineering knowledge into feature design** not only improves predictive accuracy but also enhances the interpretability of machine learning models.

---
## 📁 Repository Structure

```
├── data/
├── Data_Cleaning&Preprocessing/
├── ML_Modelling/
├── results/
├── README.md

```
---

## 👨‍🔬 Authors

* Abubakar Muhammad Danhaya
* Al-Bashir Isah Galadima
* Mahmud Muhammad Jibril 
* Aliyu Abdullahi
* Aminu Shuaibu Zarewa
* Zakiyyu Abubakar

---

## 📦 Requirements

* Python 3.8+
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib

---

## 🔍 Reproducibility

* Fixed random seed used (`random_state=100`)
* All preprocessing steps documented
* SHAP values saved for consistency

---

## 📚 Future Work

* Integration with physics-based models (FEM simulations)
* Application to structural health monitoring
* Extension to deep learning frameworks
* Hybrid physics-informed ML approaches

---
# **Thank You**