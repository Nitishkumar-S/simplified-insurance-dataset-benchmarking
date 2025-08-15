# Standard Library Imports

# TabPFN and Extensions
print("importing")
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
        AutoTabPFNClassifier, AutoTabPFNRegressor
    )
except ImportError:
    raise ImportError(
        "Warning: Could not import TabPFN / TabPFN extensions. Please run installation above and restart the session afterwards (Runtime > Restart Session)."
    )

# Data Science & Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

# Other ML Models
from catboost import CatBoostClassifier, CatBoostRegressor

# Notebook UI/Display
# from IPython.display import Markdown, display
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import Ridge
from tabpfn_extensions import AutoTabPFNRegressor


# Scikit-Learn: Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

# This transformer will be used to handle categorical features for the baseline models
column_transformer = make_column_transformer(
    (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        make_column_selector(dtype_include=["object", "category"]),
    ),
    remainder="passthrough",
)

# df2 = pd.read_csv(r"Data/car_insurance_premium_dataset_TEST.csv")
# df2.head()
# df1 = pd.read_csv(r"Data/car_insurance_premium_dataset.csv")
# df1.head()
# df = pd.concat([df1, df2], ignore_index=True)
print("Reading dataset")
dataset_path = "data/SwedishMotorInsurance.csv"
df = pd.read_csv(dataset_path)

y = df["Payment"]
X = df.drop(columns=["Payment"])

print("running model")
# Compare different machine learning models by training each one multiple times
# on different parts of the data and averaging their performance scores for a
# more reliable performance estimate
# Define models
models = [
    ("TabPFN", TabPFNRegressor(random_state=42)),
    ("AutoTabPFN", AutoTabPFNRegressor(random_state=42, device="cuda")),
    (
        "RandomForest",
        make_pipeline(
            column_transformer,  # string data needs to be encoded for model
            RandomForestRegressor(random_state=42),
        ),
    ),
    (
        "XGBoost",
        make_pipeline(
            column_transformer,  # string data needs to be encoded for model
            XGBRegressor(random_state=42),
        ),
    ),
    (
        "CatBoost",
        make_pipeline(
            column_transformer,  # string data needs to be encoded for model
            CatBoostRegressor(random_state=42, verbose=0),
        ),
    ),
    (
        "Ridge", 
        make_pipeline(
            column_transformer, 
            Ridge(alpha=1.0)
        ),
    )  # Added Ridge
]
# Calculate scores
scoring = "r2"
n_splits = 3
cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
scores = {
    name: cross_val_score(
        model, X, y, cv=cv, scoring=scoring, n_jobs=1, verbose=1
    )
    for name, model in models
}
print("completed running models")

os.makedirs("results", exist_ok=True)
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
# Plot results
df = pd.DataFrame([(k, v.mean()) for (k, v) in scores.items()], columns=["Model", "R2"])
ax = df.plot(x="Model", y="R2", kind="bar", figsize=(10, 6))
ax.set_ylim(df["R2"].min() * 0.99, df["R2"].max() * 1.01)
ax.set_title(
    f"Model Comparison - {n_splits}-fold Cross-validation \n (Variance Explained - Larger is better)"
)
# Add ROC AUC values above bars
for i, v in enumerate(df["R2"]):
    ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(f"results/model_comparison_{dataset_name}.png", dpi=300)

plt.close()
