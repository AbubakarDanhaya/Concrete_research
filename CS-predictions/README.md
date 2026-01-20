# Predicting Concrete Compressive Strength Using Machine Learning

## 📌 Overview

This repository contains the methodology, experiments, and analysis for a research study focused on predicting the **compressive strength of concrete** using multiple **machine learning (ML) models**. The study adopts a **data-driven approach**, enhanced with **ensemble learning** and **model interpretability techniques**, to achieve robust and explainable predictions.

---

## 🎯 Objectives

* Develop accurate machine learning models for predicting concrete compressive strength
* Compare the performance of individual and ensemble ML algorithms
* Improve prediction accuracy through hyperparameter optimization and ensemble learning
* Interpret model behavior using explainable AI techniques (SHAP and Partial Dependence Display)

---

## 📂 Dataset Description

* **Source**: UCI Machine Learning Repository
* **Contributor**: Prof. I-Cheng Yeh
* **Total Samples**: 1030
* **Number of Features**: 9

### Input Features

| Feature            | Unit  |
| ------------------ | ----- |
| Cement             | kg/m³ |
| Blast Furnace Slag | kg/m³ |
| Fly Ash            | kg/m³ |
| Water              | kg/m³ |
| Superplasticizer   | kg/m³ |
| Coarse Aggregate   | kg/m³ |
| Fine Aggregate     | kg/m³ |
| Age                | days  |

### Target Variable

* **Concrete Compressive Strength** (MPa)

---

## 🧹 Data Preprocessing

The dataset underwent comprehensive quality checks and cleaning steps:

* **Missing Values**: None detected
* **Duplicate Records**: 25 duplicate samples identified and removed
* **Outlier Detection**:

  * Method: Interquartile Range (IQR)
  * Outliers Identified: 94
  * Action: Excluded to minimize undue influence on model training

These preprocessing steps enhanced dataset reliability and reduced model bias.

---

## 🔀 Data Splitting

* **Training Set**: 80%
* **Testing Set**: 20%
* **Random State**: 100 (for reproducibility)

---

## ⚙️ Model Development

### Machine Learning Algorithms

Seven models were evaluated:

#### Individual Models

* AdaBoost
* Random Forest
* XGBoost
* LightGBM
* CatBoost

#### Ensemble Models

* **Voting Regressor**

  * Base Models: XGBoost, LightGBM, CatBoost

* **Stacking Regressor**

  * Base Models: XGBoost, CatBoost, Random Forest
  * Meta-Learner: RidgeCV

Ensemble models were constructed using the best-performing individual models after tuning.

---

## 🔧 Hyperparameter Optimization

* **Technique**: RandomizedSearchCV
* **Cross-Validation**: 10-fold
* **Optimization Scope**: Training dataset only

This process identified optimal model configurations while preventing data leakage.

---

## 📊 Model Evaluation

Models were evaluated on the unseen test dataset using appropriate regression performance metrics (e.g., RMSE, MAE, R²).

---

## 🔍 Model Interpretability

To enhance transparency and trust in the predictions, the following explainability techniques were employed:

* **SHAP (SHapley Additive exPlanations)**

  * Quantifies feature contributions at global and local levels

* **Partial Dependence Display (PDP)**

  * Visualizes the marginal effect of selected features on compressive strength

These methods provide insights into how input variables influence model predictions.

---

## 🛠 Tools & Technologies

* **Programming Language**: Python
* **Libraries**:

  * Scikit-learn
  * XGBoost
  * LightGBM
  * CatBoost
  * SHAP
  * NumPy, Pandas, Matplotlib, Seaborn

---

## 📈 Key Contributions

* Comprehensive comparison of individual and ensemble ML models
* Improved prediction accuracy through ensemble learning
* Integration of explainable AI for model interpretability
* Reproducible and well-structured ML workflow

---

## 📚 Reference

[1] Yeh, I.-C. (1998). *Modeling of strength of high-performance concrete using artificial neural networks*. Cement and Concrete Research.

---

## ✍️ Author

**Abubakar Muhammad Danhaya**
Civil Engineer | Machine Learning Researcher
Structural Health Monitoring & Data-Driven Modeling

---

## 📄 License

This project is intended for **academic and research purposes**. Please cite appropriately if used.
