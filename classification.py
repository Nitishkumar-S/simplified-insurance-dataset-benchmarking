# Necessary Imports for the Notebook [Running this cell required!]

# Standard Library Imports

# TabPFN and Extensions

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn_extensions import interpretability
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
import torch
import requests
import argparse
import os

# Other ML Models
from catboost import CatBoostClassifier, CatBoostRegressor

# Notebook UI/Display
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

# Travel Insurance Prediction Data
# Samples: 1.9K
# Features: 9
# Target: TravelInsurance
# link: https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data

# Download your dataset directly from GitHub

def get_data(name):
    if name == "Caravan":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/caravan-insurance-challenge.csv"
        output = "CaravanInsuranceChallenge.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.drop(columns=["ORIGIN"])
        X = df.drop(columns=["CARAVAN"])
        y = df["CARAVAN"]
    elif name == "TravelInsurance":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/TravelInsurancePrediction.csv"
        output = "TravelInsurancePrediction.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.drop(columns=["Index"])
        X = df.drop(columns=["TravelInsurance"])
        y = df["TravelInsurance"]
    elif name == "CarInsuranceClaim":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/Car_Insurance_Claim.csv"
        output = "CarInsuranceClaim.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        X = df.drop(columns=["OUTCOME"])
        y = df["OUTCOME"]
    elif name == "AutoInsuranceClaims":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/insurance_claims.csv"
        output = "insurance_claims.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.drop(columns=["_c39"])
        X = df.drop(columns=["fraud_reported"])
        y = df["fraud_reported"]
    elif name == "CarInsuranceColdCalls":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/CarInsuranceColdCalls.csv"
        output = "CarInsuranceColdCalls.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.drop(columns=["Id"])
        X = df.drop(columns=["CarInsurance"])
        y = df["CarInsurance"]
    elif name == "GermanCredit":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/germancredit.csv"
        output = "germancredit.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        X = df.drop(columns=["class"])
        y = df["class"]
    elif name == "ANUTravelClaims":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/ANUTravelClaims.csv"
        output = "ANUTravelClaims.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        X = df.drop(columns=["Status"])
        y = df["Status"]
    elif name == "PrudentialLifeInsuranceAssessment":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/PrudentialLifeInsuranceAssessment.csv"
        output = "PrudentialLifeInsuranceAssessment.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.drop(columns=["Id"])
        df = df.sample(n=10000, random_state=42)
        X = df.drop(columns=["Response"])
        y = df["Response"]
    elif name == "CarInsuranceClaimPrediction":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/CarInsuranceClaimPrediction.csv"
        output = "CarInsuranceClaimPrediction.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.sample(n=10000, random_state=42)
        X = df.drop(columns=["ClaimAmount"])
        y = df["ClaimAmount"]
    elif name == "EuropeanLapse":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/EuropeanLapse.csv"
        output = "EuropeanLapse.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.sample(n=10000, random_state=42)
        X = df.drop(columns=["Lapse"])
        y = df["Lapse"]

    return X, y


# Alternative datasets (commented for reference):

# Caravan Insurance Challenge (To predict if one will buy Caravan insurance policy)
# Samples: 9.8K
# Features: 86
# Target: CARAVAN
# link: https://www.kaggle.com/datasets/uciml/caravan-insurance-challenge
# !wget https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/caravan-insurance-challenge.csv -O CaravanInsuranceChallenge.csv
# df = pd.read_csv("CaravanInsuranceChallenge.csv")
# df = df.drop(columns=["ORIGIN"])
# X = df.drop(columns=["CARAVAN"])
# y = df["CARAVAN"]
# print(df.head())

# Car Insurance Claim Data
# Samples: 10K
# Features: 18
# Target: OUTCOME
# link: https://www.kaggle.com/datasets/sagnik1511/car-insurance-data
# !wget https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/Car_Insurance_Claim.csv -O CarInsuranceClaim.csv
# df = pd.read_csv("CarInsuranceClaim.csv")
# X = df.drop(columns=["OUTCOME"])
# y = df["OUTCOME"]
# print(df.head())

# Auto Insurance Claims Data (Fraud Detection)
# Samples: 1,000
# Features: 39
# Target: fraud_reported
# Link: https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
# Download your dataset directly from GitHub
# !wget https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/insurance_claims.csv -O insurance_claims.csv
# df = pd.read_csv("insurance_claims.csv")
# df = df.drop(columns=["_c39"])
# X = df.drop(columns=["fraud_reported"])
# y = df["fraud_reported"]
# print(df.head())

# Car Insurance Cold Calls (Customer bought car insurance or no)
# Samples: 4,000
# Features: 18
# Target: CarInsurance
# Link: https://www.kaggle.com/datasets/kondla/carinsurance
# Download your dataset directly from GitHub
# !wget https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/CarInsuranceColdCalls.csv -O CarInsuranceColdCalls.csv
# df = pd.read_csv("CarInsuranceColdCalls.csv")
# df = df.drop(columns=["Id"])
# X = df.drop(columns=["CarInsurance"])
# y = df["CarInsurance"]
# print(df.head())


# German Credit dataset(Creditworthiness indicator)
# Samples: 1000
# Features: 20
# Target: class
# Link: https://github.com/dutangc/CASdatasets/blob/master/man-md/credit.md
# Download your dataset directly from GitHub
# !wget https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/germancredit.csv -O germancredit.csv
# df = pd.read_csv("germancredit.csv")
# X = df.drop(columns=["class"])
# y = df["class"]
# print(df.head())

# ANU Corporate Travel Insurance Claims
# Samples: 2.1K
# Features: 7
# Target: Status
# link: https://datacommons.anu.edu.au/DataCommons/item/anudc:6164
# !wget https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/ANUTravelClaims.csv -O ANUTravelClaims.csv
# df = pd.read_csv("ANUTravelClaims.csv")
# X = df.drop(columns=["Status"])
# y = df["Status"]
# print(df.head())

# Prudential Life Insurance Assessment("Response" is an ordinal measure of risk that has 8 levels.)
# Samples: 59K
# Features: 127
# Target: Response
# link: https://www.kaggle.com/competitions/prudential-life-insurance-assessment
# !wget https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/PrudentialLifeInsuranceAssessment.csv -O PrudentialLifeInsuranceAssessment.csv
# df = pd.read_csv("PrudentialLifeInsuranceAssessment.csv")
# df = df.drop(columns=["Id"])
# df = df.sample(n=10000, random_state=42)
# X = df.drop(columns=["Response"])
# y = df["Response"]
# print(df.head())

# Car Insurance Claim Prediction
# Samples: 58.6K
# Features: 43
# Target: is_claim
# link: https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification
# !wget https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/CarInsuranceClaimPrediction.csv -O CarInsuranceClaimPrediction.csv
# df = pd.read_csv("CarInsuranceClaimPrediction.csv")
# df = df.sample(n=10000, random_state=42)
# X = df.drop(columns=["is_claim"])
# y = df["is_claim"]
# print(df.head())

# European lapse dataset from the direct channel(Lapse indicator)
# Samples: 23,060
# Features: 18
# Target: lapse
# Link: https://github.com/dutangc/CASdatasets/blob/master/man-md/eudirectlapse.md
# Download your dataset directly from GitHub
# !wget https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/eudirectlapse.csv -O eudirectlapse.csv
# df = pd.read_csv("eudirectlapse.csv")
# df = df.sample(n=10000, random_state=42)
# X = df.drop(columns=["lapse"])
# y = df["lapse"]
# print(df.head())

# Compare different machine learning models by training each one multiple times
# on different parts of the data and averaging their performance scores for a
# more reliable performance estimate
parser = argparse.ArgumentParser(description="Run regression models on a dataset.")
parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset CSV file")
args = parser.parse_args()

X, y = get_data(args.dataset)
# Encode target labels to classes for baselines
le = LabelEncoder()
y = le.fit_transform(y)

# Define models
models = [
    ( "TabPFN", 
        make_pipeline(
            column_transformer,
            TabPFNClassifier(random_state=42),
        ),
    ),
    ( "AutoTabPFN",
        make_pipeline(
            column_transformer,
            AutoTabPFNClassifier(random_state=42),
        ),
    ),
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

# Plot results
df = pd.DataFrame([(k, v.mean()) for (k, v) in scores.items()], columns=["Model", "ROC AUC"])
ax = df.plot(x="Model", y="ROC AUC", kind="bar", figsize=(10, 6))
ax.set_ylim(df["ROC AUC"].min() * 0.995, min(1.0, df["ROC AUC"].max() * 1.005))
ax.set_title(f"Model Comparison - {n_splits}-fold Cross-validation")

# Add ROC AUC values above bars
for i, v in enumerate(df["ROC AUC"]):
    ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig(f"results/model_comparison_{args.dataset}.png", dpi=300)

plt.close()
