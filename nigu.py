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
import requests
import argparse

# get dataset name from argument
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="caravan")
args = parser.parse_args()

# This transformer will be used to handle categorical features for the baseline models
column_transformer = make_column_transformer(
    (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        make_column_selector(dtype_include=["object", "category"]),
    ),
    remainder="passthrough",
)

"""# Classification with TabPFN <a name="classification"></a>

We will compare TabPFN's performance against other popular machine learning models: RandomForest, XGBoost, and CatBoost. The performance metric we will use is the [ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) score.

**Download dataset from github and save it locally for the session**
"""

# url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/healthInsuranceLeadPrediction.csv"
# output = "healthInsuranceLeadPrediction.csv"
# response = requests.get(url)
# with open(output, "wb") as f:
#     f.write(response.content)
# df_test = pd.read_csv("data/test_kartik.csv")
# df_train = pd.read_csv("data/train_kartik.csv")
# df = pd.concat([df_test, df_train]).sample(frac=1, random_state=42).reset_index(drop=True)
# df = df.sample(n=50, random_state=42)
# X = df.drop(columns=["Response"])
# y = df["Response"]
# print("\n\n\n\n\n\n\ndata done\n\n\n\n\n")

# Alternative datasets (commented for reference):
if args.dataset == "caravan":
    # Caravan Insurance Challenge (To predict if one will buy Caravan insurance policy)
    # Samples: 9.8K
    # Features: 86
    # Target: CARAVAN
    # link: https://www.kaggle.com/datasets/uciml/caravan-insurance-challenge
    url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/caravan-insurance-challenge.csv"
    output = "caravan.csv"
    response = requests.get(url)
    with open(output, "wb") as f:
        f.write(response.content)
    df = pd.read_csv(output)
    df = df.drop(columns=["ORIGIN"])
    X = df.drop(columns=["CARAVAN"])
    y = df["CARAVAN"]
    print(df.head())
elif args.dataset == "travel":
    # Travel Insurance Prediction Data
    # Samples: 1.9K
    # Features: 9
    # Target: TravelInsurance
    # link: https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data
    url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/TravelInsurancePrediction.csv"
    output = "TravelInsurancePrediction.csv"
    response = requests.get(url)
    with open(output, "wb") as f:
        f.write(response.content)
    df = pd.read_csv(output)
    df = df.drop(columns=["Index"])
    X = df.drop(columns=["TravelInsurance"])
    y = df["TravelInsurance"]
    print(df.head())
elif args.dataset == "insurance_claims":

    # Auto Insurance Claims Data (Fraud Detection)
    # Samples: 1,000
    # Features: 39
    # Target: fraud_reported
    # Link: https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
    # Download your dataset directly from GitHub
    url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/insurance_claims.csv"
    output = "insurance_claims.csv"
    response = requests.get(url)
    with open(output, "wb") as f:
        f.write(response.content)
    df = pd.read_csv(output)
    df = df.drop(columns=["_c39"])
    X = df.drop(columns=["fraud_reported"])
    y = df["fraud_reported"]
    print(df.head())
elif args.dataset == "car_insurance_cold_calls":

    # Car Insurance Cold Calls (Customer bought car insurance or no)
    # Samples: 4,000
    # Features: 18
    # Target: CarInsurance
    # Link: https://www.kaggle.com/datasets/kondla/carinsurance
    # Download your dataset directly from GitHub
    url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/CarInsuranceColdCalls.csv"
    output = "CarInsuranceColdCalls.csv"
    response = requests.get(url)
    with open(output, "wb") as f:
        f.write(response.content)
    df = pd.read_csv(output)
    df = df.drop(columns=["Id"])
    X = df.drop(columns=["CarInsurance"])
    y = df["CarInsurance"]
    print(df.head())
elif args.dataset == "anu_travel_claims":

    # ANU Corporate Travel Insurance Claims
    # Samples: 2.1K
    # Features: 7
    # Target: Status
    # link: https://datacommons.anu.edu.au/DataCommons/item/anudc:6164
    url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/ANUTravelClaims.csv"
    output = "ANUTravelClaims.csv"
    response = requests.get(url)
    with open(output, "wb") as f:
        f.write(response.content)
    df = pd.read_csv(output)
    X = df.drop(columns=["Status"])
    y = df["Status"]
    print(df.head())
elif args.dataset == "prudential":

    # Prudential Life Insurance Assessment("Response" is an ordinal measure of risk that has 8 levels.)
    # Samples: 59K
    # Features: 127
    # Target: Response
    # link: https://www.kaggle.com/competitions/prudential-life-insurance-assessment
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
    print(df.head())
elif args.dataset == "car_insurance_claim_prediction":

    # Car Insurance Claim Prediction
    # Samples: 58.6K
    # Features: 43
    # Target: is_claim
    # link: https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification
    url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/CarInsuranceClaimPrediction.csv"
    output = "CarInsuranceClaimPrediction.csv"
    response = requests.get(url)
    with open(output, "wb") as f:
        f.write(response.content)
    df = pd.read_csv(output)
    df = df.sample(n=10000, random_state=42)
    X = df.drop(columns=["is_claim"])
    y = df["is_claim"]
    print(df.head())
elif args.dataset == "car_insurance_claim":

    # Car Insurance Claim Data
    # Samples: 10K
    # Features: 18
    # Target: OUTCOME
    # link: https://www.kaggle.com/datasets/sagnik1511/car-insurance-data
    url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/Car_Insurance_Claim.csv"
    output = "CarInsuranceClaim.csv"
    response = requests.get(url)
    with open(output, "wb") as f:
        f.write(response.content)
    df = pd.read_csv(output)
    X = df.drop(columns=["OUTCOME"])
    y = df["OUTCOME"]
    df.head()
elif args.dataset == "eudirectlapse":

    # European lapse dataset from the direct channel(Lapse indicator)
    # Samples: 23,060
    # Features: 18
    # Target: lapse
    # Link: https://github.com/dutangc/CASdatasets/blob/master/man-md/eudirectlapse.md
    # Download your dataset directly from GitHub
    url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/eudirectlapse.csv"
    output = "eudirectlapse.csv"
    response = requests.get(url)
    with open(output, "wb") as f:
        f.write(response.content)
    df = pd.read_csv(output)
    df = df.sample(n=10000, random_state=42)
    X = df.drop(columns=["lapse"])
    y = df["lapse"]
    print(df.head())

# Compare different machine learning models by training each one multiple times
# on different parts of the data and averaging their performance scores for a
# more reliable performance estimate


def feature_selector(X, y, i):
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_enc = column_transformer.fit_transform(X)

    feature_names = X.columns
    n_features = i  #Number of features to select

    # Initialize model
    clf = TabPFNClassifier(n_estimators=1)

    # Feature selection
    sfs = interpretability.feature_selection.feature_selection(
        estimator=clf, X=X_enc, y=y, n_features_to_select=n_features, feature_names=feature_names
    )

    # Print selected features
    selected_features = [
        feature_names[i] for i in range(len(feature_names)) if sfs.get_support()[i]
    ]
    print("\nSelected features:")
    for feature in selected_features:
        print(f"- {feature}")
    return selected_features


roc_auc_scores = []
selected_features_full = []
"""**Subset of data selected based on feature importance**"""
for i in range(1, 13):
    selected_features = feature_selector(X, y, i)
    selected_features_full.append(selected_features)
    X_selected = X[selected_features]
    X_enc = column_transformer.fit_transform(X_selected)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.20, random_state=42
    )

    # Train and evaluate the TabPFN classifier
    tabpfn_classifier = TabPFNClassifier(random_state=42)
    tabpfn_classifier.fit(X_train, y_train)
    y_pred_proba = tabpfn_classifier.predict_proba(X_test)

    # Calculate the ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    roc_auc_scores.append(roc_auc)
    print(f"TabPFN ROC AUC Score: {roc_auc:.4f}")



# Create the results DataFrame
results_df = pd.DataFrame({
    "Num_Features": list(range(1, 13)),
    "Selected_Features": [", ".join(selected_features_full[:i]) for i in range(0, 12)],
    "ROC_AUC": roc_auc_scores
})


# Number of features
num_features = list(range(1, 13))  # 1 to 12 features

# Create plot
plt.figure(figsize=(8, 5))
plt.plot(num_features, roc_auc_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Features')
plt.ylabel('ROC AUC Score')
plt.title('ROC AUC vs Number of Features')
plt.xticks(num_features)
plt.grid(True)

# Save the plot
plt.savefig(f"roc_auc_vs_features_{args.dataset}.png", dpi=300, bbox_inches='tight')  # saves as PNG with high resolution

# Save to CSV
results_df.to_csv(f"roc_auc_feature_results_{args.dataset}.csv", index=False)
print(f"Results saved to 'roc_auc_feature_results_{args.dataset}.csv'")








