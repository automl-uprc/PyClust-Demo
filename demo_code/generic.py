"""This script contains a collection of methods that function on demo load or when information
to display is updated such as a meta-learner or a dataset being added to the repository. """
import gradio as gr
import json
import pandas as pd

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

def meta_learners_repository():
    """Loads and returns a dataframe that contains meta-learners and their meta-data."""
    with open("repository/meta_learners/meta-data.json", "r") as f:
        d = json.load(f)

    mldf = pd.json_normalize(d)
    mldf.columns = [".".join(x.split(".")[1:]) for x in mldf.columns]
    return mldf