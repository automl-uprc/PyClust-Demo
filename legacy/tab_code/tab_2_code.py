import sys
import gradio as gr
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import json
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append(r"C:\Users\giann\OneDrive\Έγγραφα\GitHub\cvi")
from pyclust_eval import CVIToolbox


cvi_list = list(CVIToolbox(np.array([1, 2]), np.array([1, 2])).cvi_methods_list.keys())
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99'] #used in plots


def calculate_cvi(x, y, cvi_search_type, custom_set):
    """
    Calculates cvi with PyClust-EVL
    Args:
        x (np.ndarray or pd.DataFrame): The dataset to use
        y (np.ndarray or pd.DataFrame): The clustering labels as found by any clustering algorithm
        cvi_search_type (str or list): This parameter should be 'all' or it will be ignored
        custom_set (list): The list of CVI to calculate.

    Returns:
        dict: The CVI calculated
    """
    x = np.array(x)
    y = np.array(y)

    if cvi_search_type.lower() == 'all':
        cvit = CVIToolbox(x, y)
        cvit.calculate_icvi(cvi="all")
    else:
        cvi = custom_set
        cvit = CVIToolbox(x, y)
        cvit.calculate_icvi(cvi=cvi)

    return cvit.cvi_results


def find_best_per_cvi(results_dict, data_id):
    """
    Finds the best configuration according to each CVI from given trial results
    Args:
        results_dict (dict): Trial results, referenced as master_results in main.py
        data_id (str): The dataset to search the best configurations for

    Returns:
        dict: The best configuration for each CVI
    """
    # Get all trial results in a list
    all_configs = []
    for key in results_dict[data_id]:
        all_configs += results_dict[data_id][key]

    best_config_per_cvi = {}
    for cvi in cvi_list:
        try:
            best_config_per_cvi[cvi] = max(all_configs, key=lambda x: x["cvi"][cvi])
        except Exception as e:
            print(f"Eror in finding best config: {cvi}")
            print(e)
            print(f"Eror in finding best config: {cvi}")
            continue
    print(f"Found {len(best_config_per_cvi.keys())} best configs")
    return best_config_per_cvi


def create_plots_from_es(best_config_per_cvi):
    """
    Creates two plots to serve in the UI:
    (a) histogram that displays the frequency of value for the number of clusters parameters as found per the best
        trials for each CVI.
    (b) Pie Chart used to show the frequency of the best algorithm found for each CVI.

    WARNING: This function returns None.
            Plots are saved.
    Args:
        best_config_per_cvi (dict): A dictionary that contains the trial information.
                                    Should be in the form {'algorithm': str, 'params': {}, 'labels': [], 'cvi': {}}

    Returns:
        None
    """
    print("\033[92m Creating plots from exhaustive search....")
    best_alg_count = []
    no_clusters_found_per_best = []

    for key in best_config_per_cvi:
        best_alg_count.append(best_config_per_cvi[key]["algorithm"])
        no_clusters_found_per_best.append(len(set(best_config_per_cvi[key]["labels"])))
    counter = Counter(best_alg_count)

    # ---> First Plot: Pie Chart
    pie_labels = list(counter.keys())
    pie_values = list(counter.values())

    # Create the pie chart
    plt.figure(figsize=(6, 6))

    # Title with font size and weight
    plt.title("Best Algorithm Count Per CVI", fontsize=14, fontweight='bold')

    # Enhanced pie chart
    plt.pie(
        pie_values,
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,  # Apply custom colors
        shadow=True,  # Add shadow for a 3D effect
        wedgeprops={'edgecolor': 'black'}  # Add a border around the slices
    )

    # Ensure the pie chart is circular
    plt.axis('equal')

    # Save and close the figure
    plt.savefig("best_alg_pie.png", dpi=300)  # Increase dpi for better resolution
    plt.close()

    # ---> Second Plot: Histogram
    plt.figure(figsize=(6, 6))

    # Histogram with enhanced styling
    plt.hist(
        no_clusters_found_per_best,
        bins=range(2, 22),  # Custom bin range
        edgecolor='black',  # Edge color for clarity
        color='#66b3ff',  # Custom bar color for aesthetics
        alpha=0.8  # Slight transparency for the bars
    )

    # Customize the ticks and grid
    plt.xticks(range(0, 22), fontsize=10)  # Set x-tick labels and size
    plt.yticks(fontsize=10)  # Set y-tick labels and size
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid for y-axis

    # Add labels and title with increased font sizes and bold titles
    plt.title("No Clusters Found In Best Configuration Per CVI", fontsize=14, fontweight='bold')
    plt.xlabel("No Clusters", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    # Save and close the figure with a higher resolution
    plt.savefig("no_clusters_hist.png", dpi=300)  # Save with high resolution for clarity
    plt.close()


def transform_search_space(json_input):
    """
    Utility function that takes a dictionary and expands the values to larger iterables according to [start, end, step]

    (a) list of floats = np.arrange(l[0], l[1], l[2])
    (b) list of ints = range(l[0], l[1], l[2])
    (c) single var = [var]
    Args:
        json_input (dict): The dictionary that describes parametric spaces
                            example: {"KMeans": {"no_clusters": [], ......}}

    Returns:
        dict: The updated search space
    """
    for key in json_input:
        for key_ in json_input[key]:
            if type(json_input[key][key_]) is list and type(json_input[key][key_][0]) is int:
                json_input[key][key_] = list(
                    range(json_input[key][key_][0], json_input[key][key_][1], json_input[key][key_][2]))

            elif type(json_input[key][key_]) is list and type(json_input[key][key_][0]) is float:
                json_input[key][key_] = list(
                    np.arange(json_input[key][key_][0], json_input[key][key_][1], json_input[key][key_][2]))

            elif type(json_input[key][key_]) is not list:
                json_input[key][key_] = [json_input[key][key_]]
    return json_input


def exhaustive_search(master_results, data_id, df, json_input, idx_search_type, idx_custom_set):
    """

    Args:
        master_results (dict): The dictionary that contains trial data per dataset
                                {'data id': {'alg1': [], alg2: []...}}
        data_id (str): The assigned data ID for the dataset currently active
        df (pd.DataFrame): The data provided/generated
        json_input (str): The search space defined as a string of a dict
        idx_search_type (str): Either 'all' or Custom Set
        idx_custom_set (list): The CVI to calculate if idx_search_type == Custom Set

    Returns:
        tuple: A tuple that contains visibility and content updates for Gradio components.
            (1) master_results - Content
            (2) ES success message - Content
            (3) Download Results button - Visibility
            (4) Pie-chart Image - Content
            (5) Histogram Image - Content
            (6) Best Config per CVI - Content
    """
    clustering_methods = {"KMeans": KMeans, "DBSCAN": DBSCAN}
    json_input = transform_search_space(json.loads(json_input))

    param_combinations_per_alg = {}
    for key_alg in json_input:
        # Define and iterate over parameter space for each algorithm
        param_combinations_per_alg[key_alg] = list(product(*list(json_input[key_alg].values())))
        for param_combination in param_combinations_per_alg[key_alg]:
            trial_values = {}
            params = dict(zip(json_input[key_alg].keys(), list(param_combination)))
            labels_ = clustering_methods[key_alg](**params).fit_predict(df)

            if len(set(labels_)) == 1:
                continue
            else:
                # idx_custom_set is only relevant if idx_search_type != "all", otherwise it is ignored
                cvi = calculate_cvi(df, labels_, idx_search_type, idx_custom_set)

                trial_values["algorithm"] = key_alg
                trial_values["params"] = params
                trial_values["labels"] = list([int(x) for x in list(labels_)])
                trial_values["cvi"] = cvi

                master_results[data_id][key_alg].append(trial_values)

        # Save Results
        with open("../es_search_results.json", "w") as f:
            json.dump(master_results, f)

        best_config_per_cvi = find_best_per_cvi(master_results, data_id)
        create_plots_from_es(best_config_per_cvi)

        return (master_results, gr.update(visible=True, value="<h2 style= text-align:center;>ES success</h2>"),
                gr.update(visible=True), "best_alg_pie.png",
                "no_clusters_hist.png", best_config_per_cvi)



