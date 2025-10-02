from tabpfn_extensions import interpretability
import pickle
import pandas as pd
import requests

url = "https://raw.githubusercontent.com/Nitishkumar-S/insurance-dataset/main/data/classification/CarInsuranceColdCalls.csv"
output = "data/CarInsuranceColdCalls.csv"
response = requests.get(url)
with open(output, "wb") as f:
    f.write(response.content)
df = pd.read_csv(output)
df = df.drop(columns=["Id"])
X = df.drop(columns=["CarInsurance"])
y = df["CarInsurance"]
feature_names = X.columns

with open("results/CarInsuranceColdCalls/CarInsuranceColdCalls_SHAP-IQ_shapley_interaction_values.pkl", "rb") as f:
    shap_values = pickle.load(f)
		
# interpretability.shap.plot_shap(shap_values)  # for normal SHAP method
# shap_values.plot_force(feature_names=feature_names)  # for SHAP-IQ normal shap values
shap_values.plot_upset(feature_names=feature_names)  # for SHAP-IQ interactions