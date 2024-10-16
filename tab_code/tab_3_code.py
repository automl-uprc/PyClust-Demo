import os
import sys
import seaborn as sns
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

def apply_clustering(master_results, df, algorithm, *algorithm_options):
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

    param_names = algorithm_params[algorithm]["parameters"]
    param_values = algorithm_options[algorithm_params[algorithm]["param_positions"][0]:
                                     algorithm_params[algorithm]["param_positions"][1] + 1]
    params_dict = dict(zip(param_names, param_values))
    try:
        alg = algorithm_params[algorithm]["method"](**params_dict)
        labels = alg.fit_predict(df)
        master_results[algorithm].append(params_dict)
        return master_results, gr.update(visible=True), labels
    except Exception as e:
        print(e)
        return str(e), None


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

def update_ui(selected_option):
    """
    Controls visibility of the UI. Changes according to algorithm selected (Tab 2 - 'Select Clustering Algorithm')
    Args:
        selected_option (str): Can be either KMeans or DBSCAN for now

    Returns:
        tuple: A tuple that contains visibility and content updates for Gradio components.
            (1) KMeans Param 1
            (2) KMeans Param 1
            (3) KMeans Param 1
            (4) KMeans Param 1
            (5) DBSCAN Param 1
            (6) DBSCAN Param 1
            (7) DBSCAN Param 1

    """
    if selected_option == "KMeans":
        return (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    elif selected_option == "DBSCAN":
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=True))

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


def retrieve_config(algorithm, params_dict, master_results, data_id):
    """

    Args:
        algorithm ():
        params_dict ():
        master_results ():

    Returns:

    """
    for dict_ in master_results[data_id][algorithm]:
        if dict_["params"] == params_dict:
            return dict_


def multiple_algorithm_options_to_dict(algorithm, algorithm_options):
    param_names = algorithm_params[algorithm]["parameters"]
    param_values = algorithm_options[algorithm_params[algorithm]["param_positions"][0]:
                                     algorithm_params[algorithm]["param_positions"][1] + 1]
    params_dict = dict(zip(param_names, param_values))
    return params_dict


def dimensionality_reduction(df, method, df_reduced):
    print(df)
    print(method)
    print(df_reduced)
    methods = {"T-SNE": TSNE, "PCA": PCA, "MDS": MDS}
    if df_reduced[method] is None:
        df_reduced[method] = methods[method](n_components=2).fit_transform(df)
        return df_reduced, gr.update(visible=True,
                                     value="First Trying Applying This Method, it may take a while for big "
                                           "datasets")
    else:
        return df_reduced, gr.update(visible=False)


def serve_clustering_visualizations(best_config_per_cvi, df_reduced, cvi, reduction_method, df):
    print(f'Serving clustering Visualizations')

    data = np.array(df_reduced[reduction_method])
    config = best_config_per_cvi[cvi]
    labels = config["labels"]

    # Clustering result - 1
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
    fig1 = plt.gcf()

    centroids = calculate_centroids(df, best_config_per_cvi[cvi]["labels"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(centroids.T, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Cluster Feature Heatmap")
    plt.xlabel("Clusters")
    plt.ylabel("Features")
    plt.tight_layout()  # Adjust layout to fit labels
    fig2 = plt.gcf()

    return fig1, fig2


def calculate_centroids(data, labels):
    labels = np.array(labels)
    unique_clusters = np.unique(labels)

    # Initialize array for centroids (n_clusters, n_features)
    centroids = np.zeros((len(unique_clusters), data.shape[1]))

    # Calculate the centroid for each cluster
    for i, cluster in enumerate(unique_clusters):
        cluster_points = data[labels == cluster]  # Get points in the current cluster
        centroids[i] = np.mean(cluster_points, axis=0)  # Compute the mean for each feature

    return centroids




def serve_heatmap(best_config_per_cvi, cvi, df):
    centroids = calculate_centroids(df, best_config_per_cvi[cvi]["labels"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(centroids.T, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Cluster Feature Heatmap")
    plt.xlabel("Clusters")
    plt.ylabel("Features")
    plt.tight_layout()  # Adjust layout to fit labels
    return plt




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








