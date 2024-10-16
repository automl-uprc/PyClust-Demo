import sys

import gradio as gr
import numpy as np
import pandas as pd

sys.path.append(r"./Meta-Feature-Extractors")
from clustml import MFExtractor

mfe = MFExtractor()
mf_categories = mfe.mf_categories
mf_papers = mfe.mf_papers
all_mf = [key for key in mfe.meta_features]


def update_ml_options(selected_ml):
    """
    Visibility of meta-lerner classifier parameters
    Args:
        selected_ml (str):

    Returns:

    """
    if selected_ml == "KNN":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    elif selected_ml == "DT":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)



def update_search_type_dropdown(method):
    var = {"Category": mf_categories,
           "Paper": mf_papers,
           "Custom Selection": all_mf}
    return gr.update(value="", choices=var[method])


def toggle_mf_selection_options(method):
    if method == "Custom Selection":
        return gr.update(visible=True), gr.update(visible=True)
    elif method == "All":
        return gr.update(visible=False), gr.update(visible=False)



def calculate_mf(data):
    mfe_ = MFExtractor(data)
    mfe_.calculate_mf()
    gr.Info("Meta Features Extracted Successfully!")
    return mfe_.search_mf(search_type="values"), gr.update(visible=True), gr.update(visible=True)

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