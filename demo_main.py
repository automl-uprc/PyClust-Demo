import gradio as gr
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import sys
import time
from tab_1_code import load_csv, generate_data
from tab_2_code import update_ui, apply_clustering

sys.path.append(r"C:\Users\giann\OneDrive\Έγγραφα\GitHub\cvi")
from pyclust_eval import CVIToolbox
from pyclust_eval.core._shared_processes import common_operations
from pyclust_eval.core._adg_operations import visualize_subgraph_as_tree

theme = gr.themes.Base(
    primary_hue="stone",
    secondary_hue="indigo",
    text_size="lg",
    spacing_size="sm",
).set(
    button_primary_background_fill="#FF0000",
    button_primary_background_fill_dark="#AAAAAA"
)


def calculate_cvi(x, y, cvi='all'):
    x = np.array(x.value)
    y = np.array(y.value)
    if cvi == 'all':
        start_time = time.time()
        cvit = CVIToolbox(x, y)
        cvit.calculate_icvi(cvi=cvi)
        print("its all")
        return cvit.cvi_results
    else:
        cvi = [cvi]
        start_time = time.time()
        cvit = CVIToolbox(x, y)
        cvit.calculate_icvi(cvi=cvi)
        return {cvi_calc_success_msg: gr.Markdown(visible=True,
                                                  value=f"{cvi[0]}:{cvit.cvi_results[cvi[0]]} calculated successfully in "
                                                        f"{time.time() - start_time} seconds."), }


# Create the Gradio interface
with gr.Blocks(theme=theme) as demo:
    df = gr.State()
    labels = gr.State()

    gr.Markdown("<h1 style='text-align: center;'>PyClust Demo</h1>")
    with gr.Tab('Data Loading/Generation'):
        load_data_header = gr.Markdown("<h1 style='text-align: center;'>Load Data</h1>")
        data_upload_row = gr.Row()
        generate_data_header = gr.Markdown("<h1 style='text-align: center;'>OR Generate Synthetic</h1>")
        data_generation_row = gr.Row()
        with gr.Row(visible=False) as data_success_row:
            success_msg = gr.Markdown()
        with data_upload_row:
            with gr.Column():
                upload_csv_options = gr.CheckboxGroup(["Headers", "Scale Data"])
                csv_file = gr.File(label="Upload your CSV file")
                csv_file.change(load_csv, inputs=[csv_file, upload_csv_options],
                                outputs=[load_data_header, generate_data_header, data_upload_row, data_generation_row,
                                         data_success_row,
                                         success_msg, df])
        with data_generation_row:
            with gr.Column():
                synthetic_data_method = gr.Radio(choices=["Blobs", "Moons"],
                                                 label="Choose Synthetic Data Generation Type")
                no_instances_input = gr.Number(label='No Instances')
                no_features_input = gr.Number(label='No Features')
                generate_data_btn = gr.Button('Generate Synthetic Data')
                generate_data_btn.click(generate_data, inputs=[synthetic_data_method, no_instances_input,
                                                               no_features_input],
                                        outputs=[data_upload_row, data_generation_row,
                                                 load_data_header, generate_data_header,
                                                 data_success_row, success_msg, df])
    with gr.Tab('Clustering'):
        master_results = gr.State({})
        with gr.Row():
            with gr.Column():
                gr.Markdown("<h1 style='text-align: center;'>Apply a Single Clustering Algorithm</h1>")

                algorithm_select_dropdown = gr.Dropdown(["KMeans", "DBSCAN"],
                                                        label='Select Clustering Algorithm')
                # KMeans options
                kmeans_options = {"n_clusters": gr.Slider(2, 21, value=2, step=1, interactive=True,
                                                          label="No Clusters", visible=False),
                                  "algorithm": gr.Radio(choices=["lloyd", "elkan"], label='algorithm', visible=False,
                                                        interactive=True),
                                  "max_iter": gr.Number(label="max_iter", visible=False, interactive=True, value=500),
                                  "init": gr.Radio(choices=["k-means++", "random"], label='initialization',
                                                   visible=False, interactive=True)
                                  }
                # DBSCAN options
                dbscan_options = {"eps": gr.Slider(0.1, 1, value=0.1, step=0.05, interactive=True,
                                                   label="eps", visible=False),
                                  "min_samples": gr.Slider(2, 21, value=2, step=1, interactive=True,
                                                           label="min samples", visible=False),
                                  "metric": gr.Radio(choices=["euclidean", "l1", "cosine"], label='metric',
                                                     visible=False, interactive=True)
                                  }
                all_options = [x for x in list(kmeans_options.values()) + list(dbscan_options.values())]
                # Update ui based on algorithm selected.
                algorithm_select_dropdown.change(fn=update_ui, inputs=algorithm_select_dropdown,
                                                 outputs=[kmeans_options[key] for key in kmeans_options] +
                                                         [dbscan_options[key] for key in dbscan_options])

                apply_clustering_btn = gr.Button("Apply Clustering")
                clustering_success_msg = gr.Markdown(
                    "<h1 style='text-align: center;'>Clustering Successfully applied</h1>",
                    visible=False)
                apply_clustering_btn.click(apply_clustering, inputs=[df, algorithm_select_dropdown] + all_options,
                                           outputs=[clustering_success_msg, labels])

                results = gr.State()


                def exhaustive_search(df, json_input):
                    clustering_methods = {"KMeans": KMeans,
                                          "DBSCAN": DBSCAN}

                    import json
                    from itertools import product
                    try:
                        json_input = json.loads(json_input)

                    except Exception as e:
                        print(e)
                    try:
                        for key in json_input:
                            for key_ in json_input[key]:
                                print(key, key_)
                                if type(json_input[key][key_]) is list and type(json_input[key][key_][0]) is int:
                                    print("ok_1")
                                    json_input[key][key_] = list(
                                        range(json_input[key][key_][0], json_input[key][key_][1], 1))

                                elif type(json_input[key][key_]) is list and type(json_input[key][key_][0]) is float:
                                    print("ok_2")
                                    json_input[key][key_] = list(
                                        np.arange(json_input[key][key_][0], json_input[key][key_][1], 1))
                                elif type(json_input[key][key_]) is not list:
                                    print("ok_3")
                                    json_input[key][key_] = [json_input[key][key_]]

                        param_spaces = {}
                        results_df_ = {}

                        for key in json_input:
                            results_df_[key] = pd.DataFrame(columns=list(json_input[key].keys()))
                            param_spaces[key] = list(product(*list(json_input[key].values())))

                        for key in json_input:
                            for parameter_combination in param_spaces[key]:
                                labels_ = clustering_methods[key](**dict(zip(json_input[key].keys(),
                                                                             list(parameter_combination)))).fit_predict(
                                    df.value)
                                print(labels_)
                                row_df = pd.DataFrame([dict(zip(json_input[key].keys(),
                                                                list(parameter_combination)))])
                                results_df_[key] = pd.concat([results_df_[key], row_df], ignore_index=True)

                    except Exception as e:
                        print(e)
                        return e
                    print(results_df_)
                    return results_df_


                default_search_space = """
                {
                    "KMeans": {"n_clusters": [2, 21, 1], 
                                  "tol": [0.00001, 0.0002, 10], 
                                  "algorithm": ["lloyd", "elkan"],
                                  "max_iter": 500, 
                                  "init": ["k-means++", "random"]}, 
                    "DBSCAN": {"eps": [0.01, 1, 0.05], 
                                "min_samples": [2, 21,1], 
                                "metric": ["euclidean", "l1", "cosine"]}
                }
                """
                gr.Markdown("<h1 style='text-align: center;'>Exhaustive Search</h1>")
                search_space_input = gr.Textbox(value=default_search_space,
                                                label='Set Search Space', interactive=True, lines=12)
                es_start_btn = gr.Button("Start Exhaustive Search")
                gr.Markdown("<h1 style='text-align: center;'>Clustering Parameters Tested</h1>")
                clustering_parameter_records = gr.Dataframe()
                es_start_btn.click(exhaustive_search, inputs=[df, search_space_input], outputs=results)


                def display_df(input_alg, results):
                    return results[input_alg]


                algorithm_results_select_dropdown = gr.Dropdown(["KMeans", "DBSCAN"],
                                                                label='Select Clustering Algorithm')
                algorithm_results_select_dropdown.change(display_df,
                                                         inputs=[algorithm_results_select_dropdown, results],
                                                         outputs=clustering_parameter_records)

demo.launch()
