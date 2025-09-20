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

def sequential_feature_selector(X, y, i):
    """Sequential Feature Selector using TabPFNClassifier.
    Args:
        X (pd.DataFrame): Feature dataframe.
        y (pd.Series): Target variable.
        i (int): Number of features to select.
    Returns:
        selected_features (list): List of selected feature names.
    """
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
    # selected_features = sfs.get_feature_names_out().tolist()
    
    return selected_features

if __name__ == "__main__":
    # set up logging
    logging.basicConfig(level=logging.INFO)

    # get dataset name from argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cross-sell")
    parser.add_argument("--method", type=str, default="sequential")
    args = parser.parse_args()

    # cross-sell dataset
    logging.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "cross-sell":
        df_test = pd.read_csv("data/test_kartik.csv")
        df_train = pd.read_csv("data/train_kartik.csv")
        df = pd.concat([df_test, df_train]).sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.sample(n=50, random_state=42)
        X = df.drop(columns=["Response"])
        y = df["Response"]

    # For other datasets, ***note: edit the code below to suit your dataset***
    # url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/caravan-insurance-challenge.csv"
    # output = "caravan.csv"
    # response = requests.get(url)
    # with open(output, "wb") as f:
    #     f.write(response.content)
    # df = pd.read_csv(output)
    # df = df.drop(columns=["ORIGIN"])
    # X = df.drop(columns=["CARAVAN"])
    # y = df["CARAVAN"]
    # print(df.head())

    if args.method == "sequential":
        selected_features = sequential_feature_selector(X, y, 9)
    else:
        raise ValueError("Invalid method. Please choose 'sequential'.")

    # Create the results DataFrame
    results_df = pd.DataFrame({
        args.method: selected_features
    })

    # Save the results to a CSV file
    results_df.to_csv(f"results/selected_features_{args.method}.csv", index=False)
