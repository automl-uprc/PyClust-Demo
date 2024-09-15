import sys
import time
import gradio as gr
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import json
from itertools import product
import numpy as np


def update_ui(selected_option):
    """
    Controls visibility of the UI. Changes according to algorithm selected (Tab 2 - 'Select Clustering Algorithm')
    Args:
        selected_option (str): Can be either KMeans or DBSCAN for now

    Returns: A gradio update

    """
    if selected_option == "KMeans":
        return (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    elif selected_option == "DBSCAN":
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=True))


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


def display_df(input_alg, master_results):
    """
    Displays dataframe with parameter configurations tested in the UI.
    Args:
        input_alg (str):
        master_results (dict):

    Returns:
        (pd.DataFrame) Parameter configurations tested.
    """
    return pd.DataFrame.from_records(master_results[input_alg])



