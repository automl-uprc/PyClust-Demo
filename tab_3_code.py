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

sys.path.append(r"C:\Users\giann\OneDrive\Έγγραφα\GitHub\cvi")
from pyclust_eval import CVIToolbox
from pyclust_eval.core._shared_processes import common_operations
from pyclust_eval.core._adg_operations import visualize_subgraph_as_tree

cvi_list = list(CVIToolbox(np.array([1,2]), np.array([1,2])).cvi_methods_list.keys())

def calculate_cvi(x, y, cvi_search_type, custom_set):
    x = np.array(x)
    y = np.array(y)
    if cvi_search_type.lower() == 'all':
        start_time = time.time()
        cvit = CVIToolbox(x, y)
        cvit.calculate_icvi(cvi="all")
        return cvit.cvi_results
    else:
        cvi = custom_set
        start_time = time.time()
        cvit = CVIToolbox(x, y)
        cvit.calculate_icvi(cvi=cvi)
        return cvit.cvi_results


def find_best_per_cvi(results_dict):

    best_config_per_cvi = {}
    all_configs = []

    for key in results_dict:
        for item in results_dict[key]:
            item["algorithm"] = key
        all_configs += results_dict[key]

    for cvi in cvi_list:
        try:
            best_config_per_cvi[cvi] = max(all_configs, key=lambda x: x["cvi"][cvi])
        except:
            continue

    return best_config_per_cvi


def return_best_cvi_config(best_config_per_cvi, cvi):
    """
    Returns the best configuration according to the results' dict.
    Args:
        best_config_per_cvi (dict):
        cvi (str):

    Returns:

    """
    return f"Algorithm: {best_config_per_cvi[cvi]['algorithm']}\nParameters: {best_config_per_cvi[cvi]['params']}"



def create_plots_from_es(results_dict):
    params_df = pd.DataFrame()
    cvi_df = pd.DataFrame()
    labels_list = []
    for key in results_dict:
        cvi_dict_list = [x["cvi"] for x in results_dict[key]]
        params_dict_list = [x["params"] for x in results_dict[key]]
        labels_list += [x["labels"] for x in results_dict[key]]

        cvi_df = pd.concat([cvi_df, pd.DataFrame().from_records(cvi_dict_list)])
        params_df_temp = pd.DataFrame().from_records(params_dict_list)
        params_df_temp["algorithm"] = key
        params_df = pd.concat([params_df, params_df_temp])
        print("pc -2---------------------------------------------------------")
        print(params_df)
        print(cvi_df)
        print("pc -2---------------------------------------------------------")
    params_df = params_df.reset_index()
    cvi_df = cvi_df.reset_index()

    best_alg_count = []
    no_clusters_found_per_best = []
    for cvi in list(cvi_df.columns):
        try:
            print(list(cvi_df.columns))
            print(cvi)
            print(cvi_df[cvi].idxmax())
            max_cvi_idx = cvi_df[cvi].idxmax()
            max_row = params_df.loc[max_cvi_idx]
            no_clusters_found_per_best.append(len(np.unique(labels_list[max_cvi_idx])))
            print("pc -3---------------------------------------------------------")
            print(max_row)
            print("pc -3---------------------------------------------------------")
            best_alg_count.append(max_row['algorithm'])
        except:
            continue


    print(best_alg_count)
    counter = Counter(best_alg_count)
    labels = list(counter.keys())
    values = list(counter.values())

    # First Plot: Pie Chart
    plt.figure(figsize=(6, 6))
    plt.title("Histogram of Values")
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.savefig("best_alg_pie.png")
    plt.close()

    # First Plot: Pie Chart
    plt.figure(figsize=(6, 6))
    plt.hist(no_clusters_found_per_best, bins=range(2, 22), edgecolor='black')
    plt.xticks(range(0, 22))
    # Add labels and title
    plt.title("Histogram of Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig("no_clusters_hist.png")
    plt.close()



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
        results_df_ = {}

        for key in json_input:
            results_df_[key] = pd.DataFrame(columns=list(json_input[key].keys()))
            param_spaces[key] = list(product(*list(json_input[key].values())))

        for key in json_input:
            for parameter_combination in param_spaces[key]:
                trial_values = {}

                params = dict(zip(json_input[key].keys(), list(parameter_combination)))
                labels_ = clustering_methods[key](**params).fit_predict(df)
                cvi = calculate_cvi(df, labels_, idx_search_type, idx_custom_set)

                trial_values["labels"] = list([int(x) for x in list(labels_)])
                trial_values["cvi"] = cvi
                trial_values["params"] = params

                master_results[key].append(trial_values)


        with open("es_search_results.json", "w") as f:
            json.dump(master_results, f)

        create_plots_from_es(master_results)
        best_config_per_cvi = find_best_per_cvi(master_results)

        if print(os.path.exists("es_search_results.json")):
            print("Json created Successfully!")

    except Exception as e:
        print("ES Error---------------------------------------------------------------")
        print(e)
        print("ES Error---------------------------------------------------------------")
        return e
    return (master_results, gr.update(visible=True, value="ES success"), gr.update(visible=True), "best_alg_pie.png",
            "no_clusters_hist.png", best_config_per_cvi)






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
    for dict_ in master_results[alg]:
        if dict_["params"] ==



