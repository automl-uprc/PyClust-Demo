import sys
import time
import gradio as gr
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import json
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

sys.path.append(r"C:\Users\giann\OneDrive\Έγγραφα\GitHub\cvi")
from pyclust_eval import CVIToolbox
from pyclust_eval.core._shared_processes import common_operations
from pyclust_eval.core._adg_operations import visualize_subgraph_as_tree

cvi_list = list(CVIToolbox(np.array([1, 2]), np.array([1, 2])).cvi_methods_list.keys())

print(cvi_list)
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


def calculate_cvi(x, y, cvi_search_type, custom_set):
    """
    Calculates cvi with PyClust-EVL.
    Args:
        x (np.ndarray or pd.DataFrame):
        y (np.ndarray or pd.DataFrame):
        cvi_search_type (str):
        custom_set (list):

    Returns:

    """
    x = np.array(x)
    y = np.array(y)

    if cvi_search_type.lower() == 'all':
        cvit = CVIToolbox(x, y)
        cvit.calculate_icvi(cvi="all")
        return cvit.cvi_results
    else:
        cvi = custom_set
        cvit = CVIToolbox(x, y)
        cvit.calculate_icvi(cvi=cvi)
        return cvit.cvi_results


def create_plots_from_es(best_config_per_cvi):
    best_alg_count = []
    no_clusters_found_per_best = []
    print(best_config_per_cvi)

    for key in best_config_per_cvi:
        print("labels" in best_config_per_cvi.keys())
        print("algorithm" in best_config_per_cvi.keys())
        best_alg_count.append(best_config_per_cvi[key]["algorithm"])
        no_clusters_found_per_best.append(len(set(best_config_per_cvi[key]["labels"])))

    print(best_alg_count)
    print(no_clusters_found_per_best)
    counter = Counter(best_alg_count)
    labels = list(counter.keys())
    values = list(counter.values())
    # First Plot: Pie Chart
    plt.figure(figsize=(6, 6))
    plt.title("Best Algorithm Count Per CVI")
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.savefig("best_alg_pie.png")
    plt.close()

    # Second Plot: Histogram
    plt.figure(figsize=(6, 6))
    plt.hist(no_clusters_found_per_best, bins=range(2, 22), edgecolor='black')
    plt.xticks(range(0, 22))
    # Add labels and title
    plt.title("No Clusters Found In Best Configurations Per CVI")
    plt.xlabel("No Clusters")
    plt.ylabel("Frequency")
    plt.savefig("no_clusters_hist.png")
    plt.close()


def find_best_per_cvi(results_dict):
    best_config_per_cvi = {}
    all_configs = []

    for key in results_dict:
        all_configs += results_dict[key]

    for cvi in cvi_list:
        try:
            best_config_per_cvi[cvi] = max(all_configs, key=lambda x: x["cvi"][cvi])
        except Exception as e:
            print("asdasdasd")
            print(e)
            print("asdasdasd")
            continue
    print(f"Found {len(best_config_per_cvi.keys())} best configs")
    return best_config_per_cvi


def exhaustive_search(master_results, df, json_input, idx_search_type, idx_custom_set):
    print(idx_search_type, idx_custom_set)
    clustering_methods = {"KMeans": KMeans, "DBSCAN": DBSCAN}
    json_input = json.loads(json_input)

    try:
        for key in json_input:
            for key_ in json_input[key]:
                if type(json_input[key][key_]) is list and type(json_input[key][key_][0]) is int:
                    json_input[key][key_] = list(
                        range(json_input[key][key_][0], json_input[key][key_][1], 1))
                elif type(json_input[key][key_]) is list and type(json_input[key][key_][0]) is float:
                    json_input[key][key_] = list(
                        np.arange(json_input[key][key_][0], json_input[key][key_][1], 1))
                elif type(json_input[key][key_]) is not list:
                    json_input[key][key_] = [json_input[key][key_]]

        param_spaces = {}

        for key in json_input:
            # Define and iterate over parameter space for each algorithm
            param_spaces[key] = list(product(*list(json_input[key].values())))
            for parameter_combination in param_spaces[key]:
                trial_values = {}

                # The trial should consist of {"parameters": {}, "parameters":[], "cvi": {}}
                params = dict(zip(json_input[key].keys(), list(parameter_combination)))
                labels_ = clustering_methods[key](**params).fit_predict(df)
                if len(set(labels_)) == 1:
                    continue
                else:
                    # idx_custom_set is only relevant if idx_search_type != "all"
                    cvi = calculate_cvi(df, labels_, idx_search_type, idx_custom_set)

                    trial_values["algorithm"] = key
                    trial_values["params"] = params
                    trial_values["labels"] = list([int(x) for x in list(labels_)])
                    trial_values["cvi"] = cvi

                    master_results[key].append(trial_values)

        # Save Results
        with open("es_search_results.json", "w") as f:
            json.dump(master_results, f)

        best_config_per_cvi = find_best_per_cvi(master_results)
        create_plots_from_es(best_config_per_cvi)


        if print(os.path.exists("es_search_results.json")):
            print("Json created Successfully!")

    except Exception as e:
        print("ES Error---------------------------------------------------------------")
        print(e)
        print("ES Error---------------------------------------------------------------")
        return e

    return (master_results, gr.update(visible=True, value="ES success"), gr.update(visible=True), "best_alg_pie.png",
            "no_clusters_hist.png", best_config_per_cvi)


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



