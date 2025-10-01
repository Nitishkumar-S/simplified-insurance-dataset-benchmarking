import pickle
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Any

def plot_shap(shap_values: np.ndarray) -> None:
    """Plot SHAP values for the given test data.

    This function creates several visualizations of SHAP values:
    1. Aggregated feature importances across all examples
    2. Per-sample feature importances
    3. Important feature interactions (if multiple samples provided)

    Args:
        shap_values: The SHAP values to plot, typically from get_shap_values().

    Returns:
        None: This function only produces visualizations.
    """
    import shap

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

    shap.plots.bar(shap_values=shap_values, show=False)
    plt.title("Aggregate feature importances across the test examples")
    plt.show()
    shap.summary_plot(shap_values=shap_values, show=False)
    # plot the distribution of importances for each feature over all samples
    plt.title(
        "Feature importances for each feature for each test example (a dot is one feature for one example)",
    )
    plt.show()

    most_important = shap_values.abs.mean(0).values.argsort()[-1]
    if len(shap_values) > 1:
        plot_shap_feature(shap_values, most_important)


def plot_shap_feature(
    shap_values_: Any,
    feature_name: int | str,
    n_plots: int = 1,
) -> None:
    """Plot feature interactions for a specific feature based on SHAP values.

    Args:
        shap_values_: SHAP values object containing the data to plot.
        feature_name: The feature index or name to plot interactions for.
        n_plots: Number of interaction plots to create. Defaults to 1.

    Returns:
        None: This function only produces visualizations.
    """
    import shap

    # we can use shap.approximate_interactions to guess which features
    # may interact with age
    inds = shap.utils.potential_interactions(
        shap_values_[:, feature_name],
        shap_values_,
    )

    # make plots colored by each of the top three possible interacting features
    for i in range(n_plots):
        shap.plots.scatter(
            shap_values_[:, feature_name],
            color=shap_values_[:, inds[i]],
            show=False,
        )
        plt.title(
            f"Feature {feature_name} with a color coding representing the value of ({inds[i]})",
        )
		
		
if __name__ == "__main__":

	with open("results/TravelInsurance/TravelInsurance_SHAP_shap_values.pkl", "rb") as f:
		shap_values = pickle.load(f)
		
	plot_shap(shap_values)