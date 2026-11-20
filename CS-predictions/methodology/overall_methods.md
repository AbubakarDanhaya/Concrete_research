# DATA COLLECTION AND PREPROCESSING

importing libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading DataFrame

concrete =pd.read_excel(r"c://users/danhaya/arewads/concrete_research/data/Concrete_Data.xls")

# Check for missing values
print(concrete.isna().sum())

# Check Duplicates
print(concrete.duplicated().sum())

# Drop duplicate values
concrete = concrete.drop_duplicates()
print(concrete.duplicated().sum())

# Handling outliers

sns.boxplot(data=concrete, orient='h')
plt.xlabel('Values')
plt.title("Boxplot of Features and Label(Compressive Stength)")
plt.show()

# Removing outliers

from scipy.stats import iqr
import numpy as np

# Make a copy to keep original safe
df = concrete.copy()


mask = np.ones(len(df), dtype=bool)

for col in col_list:
    data = df[col].values
    col_iqr = iqr(data)
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    lower_threshold = q1 - 1.5 * col_iqr
    upper_threshold = q3 + 1.5 * col_iqr
    
    # Update mask
    mask &= (data >= lower_threshold) & (data <= upper_threshold)

# Apply mask once
df_clean = df[mask]

print("Original shape:", concrete.shape)
print("After removing outliers:", df_clean.shape)

#create new column called water-cement ratio
df_clean['water_cement_ratio'] = df_clean['water'] / df_clean['cement']

```

# Exploratory Data Analysis (EDA)

## `DESCRIPTIVE STATISTICS` 

```python
concrete.describe()

#visualizing distributions

columns = concrete.columns

# Number of features
n = len(columns)
rows = (n // 3) + 1
cols =3

plt.figure(figsize=(15, 4*rows))

for i, col in enumerate(columns, 1):
    if col == 'water_cement_ratio':
        break
    
    plt.subplot(rows, cols, i)
    sns.histplot(concrete[col], kde=True, bins=30, color='steelblue')
    plt.title(f'{col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    

plt.tight_layout()
plt.show()

# correlation analysis
corr_matrix = concrete.corr()
corr_matrix

# correlation heatmap

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Heatmap of Concrete Dataset", fontsize=14)
plt.show()

```
# Model Development

```python
X = concrete.drop("compressive_strength", axis=1).values
y = concrete["compressive_strength"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# training functions

def randomized_search_all(X_train, y_train, X_test, y_test, n_iter=20, cv_splits=10):
    # models and parameter grids
    models_params ={
        "AdaBoost": (
            AdaBoostRegressor(random_state=100),
            {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": np.linspace(0.01, 1.0, 10)
            }
        ),
        "Random Forest" :(
            RandomForestRegressor(random_state=100),
            {
                "n_estimators": [100, 200, 300],
                "max_depth" : [5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        ),
        "XGBoost": (
            XGBRegressor(random_state=100, eval_metric='rmse'),
            {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.005, 0.1, 0.3],
                "max_depth": [3, 5, 7, 9],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0]
            }
        ),
        "LightGMB": (
            LGBMRegressor(random_state=100, verbose=1, force_col_wise=True),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [20, 30, 40]
            }
        ),
        "CatBost": (
            CatBoostRegressor(random_state=100, verbose=0, allow_writing_files=False),
            {
                "iterations": [200, 500, 800],
                "learning_rate": [0.01, 0.05, 0.1],
                "depth": [4, 6, 8, 10]
            }
        )
    }
    results ={}
    kfold = KFold(n_splits=cv_splits, shuffle = True, random_state=100)

    for name, (model, param_dist) in models_params.items():
        print(f"\nRunning RandomizedSearchCV for {name}...")

        random_search = RandomizedSearchCV(
            estimator = model,
            param_distributions = param_dist,
            n_iter = n_iter,
            scoring = 'r2',
            cv=kfold,
            verbose = 1,
            n_jobs = -1,
            random_state=100
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        results[name] = {
            "best_params": random_search.best_params_,
            "cv_r2": random_search.best_score_,
            "test_r2": r2_score(y_test, y_pred),
            "test_rmse": MSE(y_test, y_pred)**(1/2),
            "test_mae": MAE(y_test, y_pred),
            "best_model": best_model
        }
        print(f"{name} done! Best R2 CV: {random_search.best_score_:.4f}, Test R2: {results[name]['test_r2']:.4f}")

    return results

    results = randomized_search_all(X_train, y_train, X_test, y_test)

    # performance matrices and best hyperparameters

    pd.DataFrame.from_dict(results, orient='index')

    # We have best params :
best_params_xgb = {"n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 5,
                "subsample": 0.6,
                "colsample_bytree": 0.6}
           
best_params_lgbm = {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "num_leaves": 20
            }
best_params_cat = {
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 6
                }
best_params_rf =  {
                "n_estimators": 300,
                "max_depth" : 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }


# Define models using best hyperparameters
xgb = XGBRegressor(**best_params_xgb, random_state=100)
lgbm = LGBMRegressor(**best_params_lgbm, random_state=100)
cat = CatBoostRegressor(**best_params_cat, verbose=0, random_state=100)
rf = RandomForestRegressor(**best_params_rf, random_state=100)


# Voting Regressor
voting_reg = VotingRegressor(
    estimators=[('x-gradientboost', xgb), ('light-gradientboost', lgbm), ('catboost', cat)]
)
voting_reg.fit(X_train, y_train)
y_pred_voting = voting_reg.predict(X_test)
print("Voting Regressor R2:", r2_score(y_test, y_pred_voting))
print("Voting Regressor RMSE:", MSE(y_test, y_pred_voting)**(1/2))
print("Voting Regressor MAE:", MAE(y_test, y_pred_voting))


# Stacking Regressor

stacking_reg = StackingRegressor(
    estimators=[('Random_Forest', rf), ('x-gradientboost', xgb), ('catboost', cat)],
    final_estimator=RidgeCV()
)
stacking_reg.fit(X_train, y_train)
y_pred_stack = stacking_reg.predict(X_test)
print("Stacking Regressor R2:", r2_score(y_test, y_pred_stack))
print("Stacking Regressor RMSE:", MSE(y_test, y_pred_stack)**(1/2))
print("Stacking Regressor MAE:", MAE(y_test, y_pred_stack))

```
## EXPERIMENTAL VS PREDICTED
### OPTIMIZED INDIVIDUAL MODELS ONLY

```python
individual_models = [
    name for name in model_predictions
    if name not in ["Voting Regressor", "Stacking Regressor"]
]
n_models = len(individual_models)

cols = 3
rows = int(np.ceil(n_models / cols))

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(16, 10)
)

axes = np.atleast_1d(axes).flatten()

max_val = max(
    y_test.max(),
    max(model_predictions[name].max() for name in individual_models)
)

axis_max = max_val * 1.05

# Plot each individual model
for ax, name in zip(axes, individual_models):

    y_pred = model_predictions[name]

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.7,
        edgecolor='k',
        color='skyblue'
    )
    ax.plot(
        [0, axis_max],
        [0, axis_max],
        color='red',
        linestyle='-',
        linewidth=2
    )
    # -20% deviation line
    ax.plot(
        [0, axis_max],
        [0, 0.8 * axis_max],
        color='blue',
        linestyle='--',
        linewidth=1.5
    )
    # +20% deviation line
    ax.plot(
        [0, axis_max],
        [0, 1.2 * axis_max],
        color='green',
        linestyle='--',
        linewidth=1.5
    )
     # Metrics box

     metrics = model_metrics[name]

    textstr = (
        f"$R^2$ = {metrics['R²']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.2f} MPa\n"
        f"MAE = {metrics['MAE']:.2f} MPa\n"
        f"Within ±20% = {metrics['Within ±20% (%)']:.1f}%"
    )

    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment='top',
        bbox=dict(
            facecolor='white',
            alpha=0.75,
            edgecolor='gray'
        )
    )
# Titles and labels
ax.set_title(
        name,
        fontsize=12,
        fontweight='bold'
    )

    ax.set_xlabel(
        "Experimental Compressive Strength (MPa)"
    )

    ax.set_ylabel(
        "Predicted Compressive Strength (MPa)"
    )

    ax.set_xlim(
        0,
        axis_max
    )

    ax.set_ylim(
        0,
        axis_max
    )

    # remove unused subplot
    for i in range(
    n_models,
    len(axes)
):
    fig.delaxes(axes[i])
    fig.suptitle(
    "Experimental versus Predicted Compressive Strength of Optimized Individual Models",
    fontsize=15,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

```
### Plot ensemble models

```python
ensemble_models = [
    "Voting Regressor",
    "Stacking Regressor"
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5.5)
)

axes = np.atleast_1d(axes).flatten()

max_val = max(
    y_test.max(),
    max(model_predictions[name].max() for name in ensemble_models)
)

axis_max = max_val * 1.05

for ax, name in zip(axes, ensemble_models):

    y_pred = model_predictions[name]
ax.scatter(
        y_test,
        y_pred,
        alpha=0.7,
        edgecolor='k',
        color='skyblue'
    )

    # Perfect prediction line
    ax.plot(
        [0, axis_max],
        [0, axis_max],
        color='red',
        linestyle='-',
        linewidth=2
    )

     # -20% deviation line
      ax.plot(
        [0, axis_max],
        [0, 0.8 * axis_max],
        color='blue',
        linestyle='--',
        linewidth=1.5
    )
     # +20% deviation line
     ax.plot(
        [0, axis_max],
        [0, 1.2 * axis_max],
        color='green',
        linestyle='--',
        linewidth=1.5
    )
    # Metrics box
    metrics = model_metrics[name]

    textstr = (
        f"$R^2$ = {metrics['R²']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.2f} MPa\n"
        f"MAE = {metrics['MAE']:.2f} MPa\n"
        f"Within ±20% = {metrics['Within ±20% (%)']:.1f}%"
    )

    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment='top',
        bbox=dict(
            facecolor='white',
            alpha=0.75,
            edgecolor='gray'
        )
    )
     # Labels
      ax.set_title(
        name,
        fontsize=12,
        fontweight='bold'
    )

    ax.set_xlabel(
        "Experimental Compressive Strength (MPa)"
    )

    ax.set_ylabel(
        "Predicted Compressive Strength (MPa)"
    )
ax.set_xlim(
        0,
        axis_max
    )

    ax.set_ylim(
        0,
        axis_max
    )
     ax.grid(
        True,
        linestyle='--',
        alpha=0.6
    )
    # Figure title
    fig.suptitle(
    "Experimental versus Predicted Compressive Strength of Ensemble Models",
    fontsize=15,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

```

## Residual Distributions

```python
n_models = len(model_predictions)

cols = 3
rows = int(np.ceil(n_models / cols))

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(16, 14)
)

axes = axes.flatten()


# Find common residual range
all_residuals = []

for name, y_pred in model_predictions.items():
    residuals = y_test - y_pred
    all_residuals.extend(residuals)

residual_min = min(all_residuals)
residual_max = max(all_residuals)


# Plot residual distribution for each model
for ax, (name, y_pred) in zip(axes, model_predictions.items()):

    # Calculate residuals
    residuals = y_test - y_pred

    # Histogram
    ax.hist(
        residuals,
        bins=20,
        edgecolor='black',
        alpha=0.7
    )

    # Zero-error reference line
    ax.axvline(
        0,
        color='red',
        linestyle='--',
        linewidth=2
    )

    # Labels
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Residual (MPa)")
    ax.set_ylabel("Frequency")

    # Common x-axis range
    ax.set_xlim(
        residual_min,
        residual_max
    )

    ax.grid(
        True,
        linestyle='--',
        alpha=0.6
    )


# Remove empty subplots
for i in range(len(model_predictions), len(axes)):
    fig.delaxes(axes[i])


plt.tight_layout()
plt.show()

```

# Explainability
## Shap

```python
import shap
import pandas as pd
from sklearn.model_selection import train_test_split


feature_names = [
    "Cement", "Blast Furnace Slag", "Fly Ash", 
    "Water", "Superplasticizer", "Coarse Aggregate", 
    "Fine Aggregate", "Age", "water-cement ratio"
]

# Example: if X is a NumPy array
import pandas as pd
X = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


base_models = stacking_reg.named_estimators_

n_models = len(base_models)
cols = 2
rows = (n_models + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axes = axes.flatten()

for i, (name, model) in enumerate(base_models.items()):
    print(f"Explaining {name}...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    shap.plots.bar(shap_values, show=False, ax=axes[i])
    axes[i].set_title(f"SHAP Feature Importance - {name}", fontsize=11)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

```
### Shap summary plot
```python
import shap
import numpy as np

# Sample fewer rows for speed (SHAP can be heavy)
X_sample = X_train.sample(200, random_state=42)

explainer = shap.KernelExplainer(stacking_reg.predict, X_sample)
shap_values = explainer.shap_values(X_test.sample(50, random_state=42))

# Plot global importance
shap.summary_plot(shap_values, X_test.sample(50, random_state=42))

```

## Partial Dependence plot

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Check column names
print(X_train.columns.tolist())

top_features = ["Cement", "Water", "Age", "water-cement ratio", 
                "Blast Furnace Slag", "Superplasticizer", 
                "Fine Aggregate", "Coarse Aggregate", "Fly Ash"]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    if feature in X_train.columns:
        display = PartialDependenceDisplay.from_estimator(
            stacking_reg, X_train, [feature],
            kind="average",
            ax=axes[i],
            grid_resolution=50
        )
        
        ymin, ymax = axes[i].get_ylim()
        if ymin == ymax:
            axes[i].set_ylim(ymin - 0.1, ymax + 0.1)
        
        axes[i].set_title(f"PD Plot - {feature}", fontsize=11)
    else:
        axes[i].text(0.5, 0.5, f"{feature} not in X_train",
                     ha='center', va='center')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

plt.tight_layout()
plt.show()

```

# End







