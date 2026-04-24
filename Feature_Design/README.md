# 📊 Improving Machine Learning Prediction of Concrete Compressive Strength Using Engineering-Based Feature Design

## 📌 Overview

This repository presents a research-driven approach to improving the prediction of concrete compressive strength using machine learning, enhanced by **engineering-informed feature design**.

Unlike conventional data-driven approaches, this work integrates **civil engineering domain knowledge** into feature engineering to improve model accuracy, interpretability, and physical relevance.

---

## 🎯 Research Objective

The primary goal of this study is to:

> Investigate how engineering-based feature design (e.g., water–cement ratio, binder content) improves the predictive performance of machine learning models for concrete compressive strength.

---

## 📂 Dataset

* Source: UCI Machine Learning Repository
* Dataset: **Concrete Compressive Strength Dataset**

### 📊 Dataset Description

* **Total Samples:** 1030
* **Input Variables:** 8
* **Target Variable:** Compressive Strength (MPa)

#### Input Features:

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

* Data inspection:

  * Missing values
  * Duplicate records
  * Data types

* Outlier detection:

  * Interquartile Range (IQR)

Two datasets were created:

* **Dataset A:** With outliers

* **Dataset B:** Without outliers

* Feature scaling:

  * StandardScaler / MinMaxScaler (for linear models)

---

### 2. Engineering-Based Feature Design (Core Contribution)

New features were generated using domain knowledge:

* **Water–Cement Ratio (w/c)**
* **Binder Content** = Cement + Slag + Fly Ash
* **Water–Binder Ratio (w/b)**
* **Aggregate Ratio (AR)**

Two feature sets were used:

* **Raw Features**
* **Enhanced Features (Raw + Engineered)**

---

### 3. Data Splitting

* Train/Test Split:

  * 80% Training
  * 20% Testing

* Reproducibility:

  * `random_state = 42`

* Optional:

  * 5-Fold Cross Validation

---

### 4. Machine Learning Models

The following models were implemented:

* Linear Regression (Baseline)
* Random Forest Regressor
* XGBoost Regressor

---

### 5. Hyperparameter Tuning

* Methods:

  * GridSearchCV
  * RandomizedSearchCV

* Cross-validation:

  * 5-Fold CV

---

### 6. Model Evaluation Metrics

Models were evaluated using:

* **R² Score**
* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**

---

### 7. Model Interpretability

To ensure transparency and physical relevance:

* SHAP (Shapley Additive Explanations) was used to:

  * Identify key influencing features
  * Validate engineering significance

---

### 8. Visualization

The following visualizations were generated:

* Predicted vs Actual plots
* Feature importance plots
* SHAP summary plots
* Model comparison charts

---

## 📈 Results

The study demonstrates that:

* Engineering-based features significantly improve model performance
* The **water–cement ratio** is consistently the most influential parameter
* Enhanced feature sets outperform raw features across all models

---

## 🧠 Key Insight

> This research emphasizes that integrating **engineering knowledge with machine learning** leads to more accurate, interpretable, and reliable models.

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* XGBoost
* Pandas
* NumPy
* Matplotlib / Seaborn
* SHAP

---

## 📁 Repository Structure

```
├── data/
├── Data_Cleaning&Preprocessing/
├── ML_Modelling/
├── results/
├── README.md


---

## 👨‍🔬 Authors

* Abubakar Muhammad Danhaya
* Al-Bashir Isah Galadima
* Mahmud Muhammad Jibril 
* Aliyu Abdullahi
* Aminu Shuaibu Zarewa
* Zakiyyu Abubakar

---

## 📌 Future Work

* Integration with FEM-based simulation data
* Physics-informed machine learning models
* Extension to durability and long-term performance prediction

---

## ⭐ Final Note

This project bridges the gap between **data-driven models** and **engineering principles**, providing a more meaningful and practical approach to predictive modeling in civil engineering.

---

