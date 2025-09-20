# TabPFN and Extensions
try:
    from tabpfn import TabPFNClassifier                                    # TabPFNRegressor
    from tabpfn_extensions import interpretability
    # from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    #     AutoTabPFNClassifier,
    # )
except ImportError:
    raise ImportError(
        "Warning: Could not import TabPFN / TabPFN extensions. Please run installation above and restart the session afterwards (Runtime > Restart Session)."
    )

# Data Science & Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import logging

# Notebook UI/Display
from sklearn.compose import make_column_selector, make_column_transformer


# Scikit-Learn: Models
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import requests
import argparse

# This transformer will be used to handle categorical features for the baseline models
column_transformer = make_column_transformer(
    (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        make_column_selector(dtype_include=["object", "category"]),
    ),
    remainder="passthrough",
)


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