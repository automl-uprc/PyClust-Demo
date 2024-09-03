import gradio as gr
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import sys
import time
sys.path.append(r"C:\Users\giann\Documents\GitHub\PyClust-Eval")
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
                                           f"{time.time() - start_time} seconds."),}



# Function to load cluster labels
global_df = None
global_labels = None

# Create the Gradio interface
with gr.Blocks(theme=theme) as demo:
    df = gr.State()
    labels = gr.State()

    gr.Markdown("# PyClust Demo")
    with gr.Tab('Data Loading/Generation'):
        with gr.Row(visible=False) as data_success_row:
            success_msg = gr.Markdown("<h1 style='text-align: center; color: blue;'> Data Loaded Successfully!</h1>",)

        # This is the row for loading/generating the dataset
        with gr.Row() as top_row:
            with gr.Column():
                def load_csv(file):
                    global global_df
                    try:
                        if file is None:
                            return "No file uploaded", None
                        global_df = pd.read_csv(file.name)
                        return {top_row: gr.Row(visible=False),
                                data_success_row: gr.Row(visible=True),
                                success_msg: gr.Markdown(
                                    f"<h1 style='text-align: center;'> Data Loaded Successfully with "
                                    f"shape: ({global_df.shape[0]},{global_df.shape[1]}) !</h1>", ),
                                }
                    except Exception as e:
                        return str(e), None

                csv_file = gr.File(label="Upload your CSV file")
                csv_file.change(load_csv, inputs=csv_file, outputs=[top_row, data_success_row, success_msg])

            with gr.Column():
                no_instances_input = gr.Number(label='No Instances')
                no_features_input = gr.Number(label='No Features')
                generate_data_btn = gr.Button('Generate Synthetic Data')

                def generate_data(no_instances, no_features):
                    global global_df
                    x, y = make_blobs(n_samples=int(no_instances), n_features=int(no_features))
                    global_df = pd.DataFrame(x)
                    print(type(global_df))
                    print(global_df)
                    return {top_row: gr.Row(visible=False),
                            data_success_row: gr.Row(visible=True),
                            success_msg: gr.Markdown(f"<h1 style='text-align: center; color: blue;'> Data Generated Successfully with "
                                                     f"shape: ({x.shape[0]},{x.shape[1]}) !</h1>",),
                            df: gr.State(global_df),}

                generate_data_btn.click(generate_data, inputs=[no_instances_input, no_features_input],
                                        outputs=[top_row, data_success_row, success_msg, df])

        with gr.Row(visible=False) as labels_success_row:
            success_msg_labels = gr.Markdown("<h1 style='text-align: center; color: blue;'> Data Loaded Successfully!</h1>",)

        # This is the row for loading/generating the cluster labels
        with gr.Row() as labels_row:
            with gr.Column():
                def load_cluster_labels(file):
                    try:
                        labels = pd.read_csv(file.name)
                        return {labels_row: gr.Row(visible=False),
                                labels_success_row: gr.Row(visible=True),
                                success_msg_labels: gr.Markdown(
                                    f"<h1 style='text-align: center; color: blue;'> Cluster Labels Loaded Successfully with "
                                    f"no clusters: ({np.unique(labels[labels.columns[0]])}) !</h1>", ), }
                    except Exception as e:
                        return str(e), None
                csv_labels_file = gr.File(label="Upload your Cluster Labels (CSV)")
                csv_labels_file.change(load_cluster_labels, inputs=csv_labels_file, outputs=[labels_row,labels_success_row,
                                                                                      success_msg_labels])


            with gr.Column():
                def generate_clusters(n_clusters):
                    global global_labels
                    try:
                        print("asd")
                        print(global_df)
                        kmeans = KMeans(n_clusters=n_clusters)
                        global_labels = kmeans.fit_predict(global_df)
                        print(np.unique(global_labels))
                        return {labels_row: gr.Row(visible=False),
                                labels_success_row: gr.Row(visible=True),
                                success_msg_labels: gr.Markdown(
                                    f"<h1 style='text-align: center; color: blue;'> Dataset partitioned into {n_clusters} clusters succesfully!"
                                    f"</h1>", ),
                                labels: gr.State(global_labels)}
                    except Exception as e:
                        return str(e), None


                def update_ui(selected_option):
                    if selected_option == "KMeans":
                        return gr.update(visible=True), gr.update(visible=False)
                    elif selected_option == "DBSCAN":
                        return gr.update(visible=False), gr.update(visible=True)


                algorithm_select_dropdown = gr.Dropdown(["KMeans", "DBSCAN"], label='Select Clustering Algorithm')
                temp_slider_kmeans = gr.Slider(
                    2, 20,
                    value=2,
                    step=1,
                    interactive=True,
                    label="No Clusters",
                    visible=False
                )
                temp_slider_dbscan = gr.Slider(
                    0.1, 1,
                    value=0.1,
                    step=0.05,
                    interactive=True,
                    label="eps",
                    visible=False
                )
                generate_cluster_labels_btn = gr.Button('Generate Cluster Labels')
                generate_cluster_labels_btn.click(generate_clusters, inputs=temp_slider_kmeans,
                                        outputs=[labels_row, labels_success_row, success_msg_labels, labels])

                algorithm_select_dropdown.change(fn=update_ui, inputs=algorithm_select_dropdown,
                                                 outputs=[temp_slider_kmeans, temp_slider_dbscan])

    with gr.Tab('CVI Calculation'):
        with gr.Row() as single_cvi_row:
            with gr.Column():
                cvi_select = gr.Textbox(label="Select CVI:")
                calculate_single_cvi_btn = gr.Button("Calculate CVI")
                cvi_calc_success_msg = gr.Markdown(visible=False)
                execution_plan = gr.Image(interactive=False, label='Execution Plan')
                subprocess_dropdown = gr.Dropdown(choices=[x for x in common_operations], label="Visualize Subprocess")
                subprocess_dropdown.change(visualize_subgraph_as_tree, inputs=subprocess_dropdown, outputs=execution_plan)
                calculate_single_cvi_btn.click(fn=calculate_cvi, inputs=[df, labels, cvi_select,
                                                                         cvi_calc_success_msg], outputs=[cvi_calc_success_msg])

        with gr.Row() as all_cvi_row:
            with gr.Column():
                calculate_btn = gr.Button("Calculate all CVI")
                all_cvi_json = gr.Json()
                calculate_btn.click(calculate_cvi, inputs=[df, labels], outputs=[all_cvi_json])
demo.launch()
