"""
Contains all the demo_code used in the first tab of PyClust Demo, used for loading or generating data.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs, make_moons
import gradio as gr


def change_data_update_visibility():
    """
    Controls visibility when the user wants to change the current working dataset.
    Returns:
        tuple: A tuple that contains visibility and content updates for Gradio components.
            (1) Data loading Row - Visibility
            (2) Data Generation Row - Visibility
            (3) data id Textbox - Visibility
            (4) data success Row - Visibility
    """
    return (gr.update(visible=True),
            gr.update(visible=False))


def load_csv(file, csv_options, data_id):
    """
    Loads a CSV dataset and updates UI.
    Args:
        file (file-like object or None): The path to load dataset from
        csv_options (list): a list which may contain one or more of Headers, Scale Data (MinMax)
        label_id (str): The id to assign to the dataset being uploaded.

    Returns:
        tuple: A tuple that contains visibility and content updates for Gradio components.
            (1) Data loading Row - Visibility
            (2) Data Generation Row - Visibility
            (3) data id Textbox - Visibility
            (4) data success Row - Visibility
            (5) success message Markdown - Value
            (6) dataframe State - Value
            (7) Subtitle Markdown - Value
    """
    print(f"Received:\n1)File: {file}\n2)csv options:{csv_options}\n3)Data ID:{data_id}", flush=True)
    if 'Headers' in csv_options:
        headers = 0
    else:
        headers = None
    try:
        df_ = pd.read_csv(file.name, header=headers)
        if 'Scale Data (MinMax)' in csv_options:
            ms = MinMaxScaler()
            df_ = ms.fit_transform(df_)

        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=True),
                f"<h1 style='text-align: center; color: blue;'> Data Loaded Successfully with shape: ({df_.shape[0]},"
                f"{df_.shape[1]}) !</h1>",
                df_,
                data_id,
                f"<h2 style='text-align: center;'>Working on Dataset with ID: {data_id}.</h2>")

    except Exception as e:
        return str(e), None


def generate_data(synthetic_method, no_instances, no_features, data_id):
    """
    Generates synthetic data based on user's preferences.
    Args:
        synthetic_method (str): Can be one of Blobs or Moons
        no_instances (int): Number of rows to generate
        no_features (int): Number of features to generate
        data_id (str): The assigned dataset label, provided by the user

    Returns:
        tuple: A tuple that contains visibility and content updates for Gradio components.
            (1) Data loading Row - Visibility
            (2) Data Generation Row - Visibility
            (3) data id Textbox - Visibility
            (4) data success Row - Visibility
            (5) success message Markdown - Value
            (6) dataframe State - Value
            (7) Subtitle Markdown - Value

    """
    x = None
    if synthetic_method == 'Blobs':
        x, y = make_blobs(n_samples=int(no_instances), n_features=int(no_features))
    elif synthetic_method == 'Moons':
        x, y = make_moons(n_samples=no_instances)
    df_ = pd.DataFrame(x)

    return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=True),
            f"<h1 style='text-align: center; color: blue;'> Data Loaded Successfully with shape: ({df_.shape[0]},"
            f"{df_.shape[1]}) !</h1>",
            df_,
            data_id,
            f"<h2 style='text-align: center; color: blue;'>Working on Dataset with ID: {data_id}.</h2>")
