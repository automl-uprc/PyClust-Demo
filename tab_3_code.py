import os
import sys

import gradio as gr
import numpy as np
import time
from sklearn.cluster import KMeans, DBSCAN
import json
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
import ast
import matplotlib.colors as mcolors




algorithm_params = {"KMeans": {
    "parameters": ["n_clusters", "algorithm", "max_iter", "init"],
    "param_positions": [0, 4],
    "method": KMeans
},
    "DBSCAN": {
        "parameters": ["eps", "min_samples", "metric"],
        "param_positions": [4, 7],
        "method": DBSCAN
    }}


def select_option(tickbox_1, tickbox_2, best_config_browse):
    """
    Function to control configuration selection in the UI. There can only be one tickbox selected or neither.
    Args:
        tickbox_1 (bool):
        tickbox_2 (bool):
        best_config_browse ():

    Returns:

    """
    if tickbox_1 is False and tickbox_2 is False:
        return best_config_browse["labels"], gr.update(value=False), gr.update(value=False)
    else:
        return best_config_browse["labels"], gr.update(value=False), gr.update(value=True)


def statistics_per_cluster(df, labels):
    df["Label"] = labels
    all_stats = []

    # Loop through each cluster
    for label, df_cluster in df.groupby("Label"):
        df_cluster = df_cluster.drop(columns=["Label"])  # Remove the label column for stats calculation

        # Compute statistics for the current cluster, per feature
        stats_df = pd.DataFrame({
            "mean": df_cluster.mean(),
            "std": df_cluster.std(),
            "min": df_cluster.min(),
            "max": df_cluster.max(),
            "median": df_cluster.median(),
            "var": df_cluster.var(),
            "skew": df_cluster.skew(),
            "kurt": df_cluster.kurtosis(),
        })

        # Add a column to indicate the cluster label
        stats_df["Cluster"] = label

        # Add the feature names as an index
        stats_df["Feature"] = stats_df.index

        # Append the stats for this cluster to the list
        all_stats.append(stats_df)

    # Concatenate all cluster stats into a single DataFrame
    final_stats_df = pd.concat(all_stats).reset_index(drop=True)

    # Return the combined statistics as a single DataFrame
    return gr.Dataframe(final_stats_df)


def retrieve_config(algorithm, params_dict, master_results):
    """

    Args:
        algorithm ():
        params_dict ():
        master_results ():

    Returns:

    """
    for dict_ in master_results[algorithm]:
        if dict_["params"] == params_dict:
            return dict_


def multiple_algorithm_options_to_dict(algorithm, algorithm_options):
    param_names = algorithm_params[algorithm]["parameters"]
    param_values = algorithm_options[algorithm_params[algorithm]["param_positions"][0]:
                                     algorithm_params[algorithm]["param_positions"][1] + 1]
    params_dict = dict(zip(param_names, param_values))
    return params_dict


def dimensionality_reduction(df, method, df_reduced):
    methods = {"T-SNE": TSNE, "PCA": PCA, "MDS": MDS}
    if df_reduced[method] is None:
        df_reduced[method] = methods[method](n_components=2).fit_transform(df)
        return df_reduced, gr.update(visible=True,
                                     value="First Trying Applying This Method, it may take a while for big "
                                           "datasets")
    else:
        return df_reduced, gr.update(visible=False)


def serve_clustering_visualization(master_results, df, df_reduced, method, option_1, option_2, config, algorithm,
                                   *params):
    """

    Args:
        master_results ():
        df_reduced ():
        method ():
        option_1 (bool): config by best of ES
        option_2 (bool): manual configuration
        config ():
        algorithm ():
        *params ():

    Returns:

    """

    data = np.array(df_reduced[method])
    if option_1:
        config = config.split("\n")
        algorithm = config[0].split(":")[1].replace(" ", "")
        parameters = ast.literal_eval(config[1].split("Parameters:")[1])

        # Retrieve full config
        full_config = retrieve_config(algorithm=algorithm, params_dict=parameters, master_results=master_results)
        labels = full_config["labels"]
        print(labels)

    cmap = plt.cm.get_cmap("viridis", len(np.unique(labels)))
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(np.unique(labels)), 1), cmap.N)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, norm=norm, s=50)
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title("2D Dataset with Colors Based on Labels")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.savefig("clusters_visualized.png")
    return "clusters_visualized.png"








def return_best_cvi_config(best_config_per_cvi, cvi):
    """
    Returns the best configuration according to the results' dict.
    Args:
        best_config_per_cvi (dict):
        cvi (str):

    Returns:

    """
    return (best_config_per_cvi[cvi],
            f"Algorithm: {best_config_per_cvi[cvi]['algorithm']}\nParameters: {best_config_per_cvi[cvi]['params']}")








def find_best(cvi, results_dict):
    results_pd = pd.DataFrame()
    for key in results_dict:
        cvi_dict = [x["cvi"] for x in results_dict[key]]
        alg_pd = pd.DataFrame().from_records(cvi_dict)
        alg_pd["algorithm"] = str(key)
        results_pd = pd.concat([results_pd, alg_pd])
    results_pd = results_pd.reset_index()

    print(results_pd)
    max_row = results_pd.loc[results_pd[cvi].idxmax()]
    max_row = max_row.replace([np.inf, -np.inf], ['Infinity', '-Infinity']).fillna('NaN')
    print(max_row)
    return json.dumps(max_row.to_dict())


def check_if_config_exists(master_results, algorithm, *algorithm_options):
    param_names = algorithm_params[algorithm]["parameters"]
    param_values = algorithm_options[algorithm_params[algorithm]["param_positions"][0]:
                                     algorithm_params[algorithm]["param_positions"][1] + 1]
    params_dict = dict(zip(param_names, param_values))

    for dict_ in master_results[algorithm]:
        if dict_["params"] == params_dict:
            return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                    gr.update(interactive=True))

    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(interactive=False)
