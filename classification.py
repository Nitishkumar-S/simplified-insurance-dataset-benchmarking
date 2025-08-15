# Standard Library Imports

# TabPFN and Extensions

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
        AutoTabPFNClassifier,
    )
except ImportError:
    raise ImportError(
        "Warning: Could not import TabPFN / TabPFN extensions. Please run installation above and restart the session afterwards (Runtime > Restart Session)."
    )

# Data Science & Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Other ML Models
from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn.compose import make_column_selector, make_column_transformer


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

dataset_path = "/content/insurance.csv"
df = pd.read_csv(dataset_path)

y = df["Insurance Premium ($)"]
X = df.drop(columns=["Insurance Premium ($)"])


# Compare different machine learning models by training each one multiple times
# on different parts of the data and averaging their performance scores for a
# more reliable performance estimate

# Encode target labels to classes for baselines
le = LabelEncoder()
y = le.fit_transform(y)

# Define models
models = [
    ("TabPFN", TabPFNClassifier(random_state=42)),
    ("AutoTabPFN", AutoTabPFNClassifier(random_state=42, device="cuda")),
    (
        "RandomForest",
        make_pipeline(
            column_transformer,  # string data needs to be encoded for model
            RandomForestClassifier(random_state=42),
        ),
    ),
    (
        "XGBoost",
        make_pipeline(
            column_transformer,  # string data needs to be encoded for model
            XGBClassifier(random_state=42),
        ),
    ),
    (
        "CatBoost",
        make_pipeline(
            column_transformer,  # string data needs to be encoded for model
            CatBoostClassifier(random_state=42, verbose=0),
        ),
    ),
]

# Calculate scores
n_splits = 3
cv = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
scoring = "roc_auc_ovr" if len(np.unique(y)) > 2 else "roc_auc"
scores = {
    name: cross_val_score(
        model, X, y, cv=cv, scoring=scoring, n_jobs=1, verbose=1
    )
    for name, model in models
}


os.makedirs("results", exist_ok=True)
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

# Plot results
df = pd.DataFrame([(k, v.mean()) for (k, v) in scores.items()], columns=["Model", "ROC AUC"])
ax = df.plot(x="Model", y="ROC AUC", kind="bar", figsize=(10, 6))
ax.set_ylim(df["ROC AUC"].min() * 0.995, min(1.0, df["ROC AUC"].max() * 1.005))
ax.set_title(f"Model Comparison - {n_splits}-fold Cross-validation")

# Add ROC AUC values above bars
for i, v in enumerate(df["ROC AUC"]):
    ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(f"results/model_comparison_{dataset_name}.png", dpi=300)

plt.close()
