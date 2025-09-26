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
import pickle 

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
    output = "data/"
    if name == "Caravan":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/caravan-insurance-challenge.csv"
        output += "CaravanInsuranceChallenge.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.drop(columns=["ORIGIN"])
        X = df.drop(columns=["CARAVAN"])
        y = df["CARAVAN"]
    elif name == "TravelInsurance":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/TravelInsurancePrediction.csv"
        output += "TravelInsurancePrediction.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.drop(columns=["Index"])
        X = df.drop(columns=["TravelInsurance"])
        y = df["TravelInsurance"]
    elif name == "CarInsuranceClaim":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/Car_Insurance_Claim.csv"
        output += "CarInsuranceClaim.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        X = df.drop(columns=["OUTCOME"])
        y = df["OUTCOME"]
    elif name == "AutoInsuranceClaims":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/insurance_claims.csv"
        output += "insurance_claims.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.drop(columns=["_c39"])
        X = df.drop(columns=["fraud_reported"])
        y = df["fraud_reported"]
    elif name == "CarInsuranceColdCalls":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/CarInsuranceColdCalls.csv"
        output += "CarInsuranceColdCalls.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.drop(columns=["Id"])
        X = df.drop(columns=["CarInsurance"])
        y = df["CarInsurance"]
    elif name == "GermanCredit":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/germancredit.csv"
        output += "germancredit.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        X = df.drop(columns=["class"])
        y = df["class"]
    elif name == "ANUTravelClaims":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/ANUTravelClaims.csv"
        output += "ANUTravelClaims.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        X = df.drop(columns=["Status"])
        y = df["Status"]
    elif name == "PrudentialLifeInsuranceAssessment":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/PrudentialLifeInsuranceAssessment.csv"
        output += "PrudentialLifeInsuranceAssessment.csv"
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
        output += "CarInsuranceClaimPrediction.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.sample(n=10000, random_state=42)
        X = df.drop(columns=["ClaimAmount"])
        y = df["ClaimAmount"]
    elif name == "EuropeanLapse":
        url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/EuropeanLapse.csv"
        output += "EuropeanLapse.csv"
        response = requests.get(url)
        with open(output, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(output)
        df = df.sample(n=10000, random_state=42)
        X = df.drop(columns=["Lapse"])
        y = df["Lapse"]

    feature_names = X.columns
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = column_transformer.fit_transform(X)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    # set up logging
    logging.basicConfig(level=logging.INFO)

    # get dataset name from argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--method", type=str, default="all")
    args = parser.parse_args()

    dataset_list = [
    "Caravan",
    "TravelInsurance",
    "CarInsuranceClaim",
    "AutoInsuranceClaims",
    "CarInsuranceColdCalls",
    "GermanCredit",
    "ANUTravelClaims",
    "PrudentialLifeInsuranceAssessment",
    "CarInsuranceClaimPrediction",
    "EuropeanLapse"
    ]

    method_list = ["SHAP", "SHAP-IQ", "PDP", "ICE"]

    # Parse dataset argument
    if args.dataset == "all":
        datasets_to_use = dataset_list
    else:
        datasets_to_use = [d.strip() for d in args.dataset.split(";")]

    # Parse method argument
    if args.method == "all":
        methods_to_use = method_list
    else:
        methods_to_use = [m.strip() for m in args.method.split(";")]

    for name in datasets_to_use:
        logging.info(f"Processing dataset: {name}")

        # Load dataset
        X_train, X_test, y_train, y_test, feature_names = get_data(name)

        # Initialize and train model
        clf = TabPFNClassifier()
        clf.fit(X_train, y_train)

        for method in methods_to_use:
            logging.info(f"Running method: {method} on dataset: {name}")
            filename = f"results/{name}/{name}_{method}"

            if method == "SHAP":
                # Calculate SHAP values
                shap_values = interpretability.shap.get_shap_values(
                    estimator=clf,
                    test_x=X_test,
                    attribute_names=feature_names,
                    algorithm="permutation",
                )
                with open(f"{filename}_shap_values.pkl", "wb") as f:
                    pickle.dump(shap_values, f)
                
                #with open(f"{filename}_shap_values.pkl", "rb") as f:
                #    shap_values_loaded = pickle.load(f)
                # Create visualization
                #fig = interpretability.shap.plot_shap(shap_values)
                # Save figure with dataset + method name
                #fig.savefig(filename, dpi=300, bbox_inches="tight")
                #plt.close(fig)

            elif method == "SHAP-IQ":
                n_model_evals = 100
                x_explain = X_test[0]
                # Get a TabPFNExplainer
                explainer = interpretability.shapiq.get_tabpfn_explainer(
                    model=clf,
                    data=X_train,
                    labels=y_train,
                    index="SV",  # SV: Shapley Value (like in shap)
                    verbose=True,  # show a progress bar during explanation
                )

                # Get shap values
                logging.info("Calculating SHAP values...")
                shapley_values = explainer.explain(x=x_explain, budget=n_model_evals)
                with open(f"{filename}_shapley_values.pkl", "wb") as f:
                    pickle.dump(shapley_values, f)

                # plot the force plot
                #shapley_values.plot_force(feature_names=feature_names)

                # Get an Shapley Interaction Explainer (here we use the Faithful Shapley Interaction Index)
                explainer = interpretability.shapiq.get_tabpfn_explainer(
                    model=clf,
                    data=X_train,
                    labels=y_train,
                    index="FSII",  # SV: Shapley Value, FSII: Faithful Shapley Interaction Index
                    max_order=1,  # maximum order of the Shapley interactions (2 for pairwise interactions)
                    verbose=True,  # show a progress bar during explanation
                )

                # Get shapley interaction values
                logging.info("Calculating Shapley interaction values...")
                shapley_interaction_values = explainer.explain(x=x_explain, budget=n_model_evals)

                with open(f"{filename}_shapley_interaction_values.pkl", "wb") as f:
                    pickle.dump(shapley_interaction_values, f)

                # Plot the upset plot for visualizing the interactions
                #shapley_interaction_values.plot_upset(feature_names=feature_names)

            elif method == "PDP":
                # 1D PD for the first 3 features + a 2D interaction plot
                disp = interpretability.pdp.partial_dependence_plots(
                    estimator=clf,
                    X=X_test,
                    features = [0, 1, 2, (0, 3)],#list(range(len(feature_names))) + [(1, len(feature_names)-1)],
                    grid_resolution=30,
                    kind="average",
                    target_class=1,
                )
                disp.figure_.suptitle("Partial dependence")

                plt.savefig(f"{filename}.png")

            elif method == "ICE":
                # 1D PD for the first 3 features + a 2D interaction plot
                disp = interpretability.pdp.partial_dependence_plots(
                    estimator=clf,
                    X=X_test,
                    #features = list(range(len(feature_names))) + [(1, len(feature_names)-1)],
                    features = [0, 1, 2, (0, 3)],
                    grid_resolution=30,
                    kind="individual",
                    target_class=1,
                )
                disp.figure_.suptitle("Partial dependence")

                plt.savefig(f"{filename}.png")

            else:
                raise ValueError(f"Invalid method: {method}")

            logging.info(f"Saved {method} plot as {filename}")