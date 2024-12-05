"""This script contains a collection of methods that function on demo load or when information
to display is updated such as a meta-learner or a dataset being added to the repository. """
import os

import gradio as gr
import json
import pandas as pd


def meta_learners_repository(ml_metadata_path="repository/meta_learners/meta-data.json"):
    """
    Loads and returns a dataframe that contains meta-learners and their meta-data.

    Args:
        ml_metadata_path (str): Path to the meta-learners meta-data. Must be a valid JSON file

    Returns:
        (pd.DataFrame): A pandas dataframe that contains meta-data of the trained meta-learners.
    """

    with open(ml_metadata_path, "r") as f:
        d = json.load(f)

    mldf = pd.DataFrame()
    for k in d.keys():
        row = pd.json_normalize(d[k])
        row.insert(0, "ML-ID", k)
        mldf = pd.concat([mldf, row])

    return mldf


def on_startup_read_no_datasets():
    """
    Retrieves the number of datasets present in the repository. Currently based on meta-features files present. Does
    take into account subfolders.

    Returns:
        (int): Number of datasets
    """
    file_count = 0
    for root, dirs, files in os.walk("repository/meta_features"):
        file_count += len(files)
    return file_count


def update_cache_results(data_id_, master_results_):
    master_results_[data_id_] = {"KMeans": [], "DBSCAN": []}
    return master_results_


def on_operations_change(state):
    print("triggered")
    operation_results_success = {
        "meta-features-extraction": "<h2 style='text-align: right; color:#3ebefe;'>Meta-Features: ✅</h2>",
        "configurations-search": "<h2 style='text-align: left; color:#3ebefe;'>Model Search: ✅</h2>",
        "results_saved_to_repo": False}

    operation_results_failure = {
        "meta-features-extraction": "<h2 style='text-align: right; color:#3ebefe;'>Meta-Features: ❌</h2>",
        "configurations-search": "<h2 style='text-align: left; color:#3ebefe;'>Model Search: ❌</h2>",
        "results_saved_to_repo": False}

    if state["meta-features-extraction"]:
        x_1 = operation_results_success["meta-features-extraction"]
    else:
        x_1 = operation_results_failure["meta-features-extraction"]

    if state["configurations-search"]:
        x_2 = operation_results_success["configurations-search"]
    else:
        x_2 = operation_results_failure["configurations-search"]

    if state["meta-features-extraction"] and state["configurations-search"]:
        x_3 = gr.update(visible=True)
    else:
        x_3 = gr.update(visible=False)

    if state["results_saved_to_repo"]:
        x_4 = "<h2 style='text-align: right; color:#3ebefe;'>Dataset in Repository: ✅</h2>"
        x_5 = gr.update(visible=False)
    else:
        x_4 = "<h2 style='text-align: right; color:#3ebefe;'>Dataset in Repository: ❌</h2>"
        x_5 = gr.update(visible=True)

    return x_1, x_2, x_3, x_4, x_5


def on_df_load(df):
    """
    Updates on UI/cache after dataset is provided.
    Returns:
        (tuple):
            (0) df_features_more_than_two: Bool
            (1) Grid Search Tab: Visibility
            (2) Clustering Exploration Tab: Visibility
            (3) Not Loaded Message: Visibility
    """
    if df.shape[1] > 2:
        needs_reduction = True
    else:
        needs_reduction = False

    return needs_reduction, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)


def df_needs_dimred(df_needs_dim_red):
    if df_needs_dim_red:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)





def control_data_visibility(data_method):
    """
    Changes layout for data upload/generation section
    Args:
        data_method (str): will be either Upload, Generate or None

    Returns:
        - 1 visibility update: Data upload Column
        - 2 visibility update: Data Generation Column
    """

    if data_method == "Upload":
        return gr.update(visible=False), gr.update(visible=True)

    elif data_method == "Generate":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False)
