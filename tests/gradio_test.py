import gradio
import gradio.components.chatbot

from tab_code.tab_1_code import *
from tab_code.tab_2_code import *
from tab_code.tab_3_code import (serve_clustering_visualizations,
                                 return_best_cvi_config, statistics_per_cluster,
                                 dimensionality_reduction)
from tab_code.tab_4_code import update_search_type_dropdown, calculate_mf, update_ml_options, toggle_mf_selection_options
from css import custom_css

css = """
#my-textbox, #my-dropdown {
    height: 120px;  /* Set a common height */
    font-size: 16px; /* Adjust font size if needed */
#fixed_height_col {
    height: 6000px;  /* Set the specific height */
    overflow: auto; /* Optional: Ensure content is scrollable if it overflows */
}
"""

logo_path = 'clustering_framework_logo.jpg'


def update_cache_results(data_id_, master_results_):
    master_results_[data_id_] = {"KMeans": [], "DBSCAN": []}
    return master_results_


def control_visibility_1(data_method):
    if data_method == "Upload":
        return gr.update(visible=False), gradio.update(visible=True)
    elif data_method == "Generate":
        return gr.update(visible=True), gradio.update(visible=False)


def enable_tabs_after_df():
    return (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))


# Create the Gradio interface
with gr.Blocks(css=css) as demo:
    # Cache data
    df = gr.State()

    data_id = gr.State()

    df_reduced = gr.State({"MDS": None, "PCA": None, "T-SNE": None})
    labels = gr.State()
    best_config_per_cvi = gr.State({})
    best_config_browse = gr.State()
    custom_config_browse = gr.State()

    # Models tested are temporarily being saved in memory
    master_results = gr.State({})
    cache_results = gr.State({})
    data_id.change(update_cache_results, inputs=[data_id, master_results], outputs=[master_results])

    # <-----------------------------Demo Titles ✓-------------------------------------------------------------->

    demo_title = gr.Markdown("<h1 style='text-align: center; overflow: hidden; font-size: 50px;'>PyClust Demo</h1>")

    demo_title_dataset_loaded = gr.Markdown("<h2 style='text-align: center; color:red;'>No dataset loaded!</h2>")

    # <-----------------------------Data Loading ✓-------------------------------------------------------------->
    with gr.Tab('Data Loading/Generation'):
        # Content
        data_method_row = gr.Row(visible=True)
        data_upload_row = gr.Row(visible=False)
        data_generation_row = gr.Row(visible=False)

        with data_method_row:
            data_id_txtbox = gr.Textbox(label="Assign dataset ID", elem_classes="center-label")
            data_method = gr.Radio(choices=["Upload", "Generate"], label="Select How to Provide Data",
                                   elem_classes="center-radio")

        data_method.change(control_visibility_1, inputs=data_method, outputs=[data_generation_row, data_upload_row])

        # Only visible after dataset is loaded or generated.
        with gr.Column(visible=False) as data_success_row:
            success_msg = gr.Markdown()
            change_data_btn = gr.Button("Change Dataset?")
            change_data_btn.click(change_data_update_visibility, outputs=[data_method_row, data_success_row])

        # - Data Upload Column
        with data_upload_row:
            with gr.Column():
                load_data_header = gr.Markdown(
                    "<h1 style='text-align: center;'> \u27A1 Load Data ([.csv])</h1>")
                upload_csv_options = gr.CheckboxGroup(["Headers", "Scale Data (MinMax)"],
                                                      label="Preprocessing Options")
                csv_file = gr.File(label="Upload your CSV file")
                csv_file.change(load_csv, inputs=[csv_file, upload_csv_options, data_id_txtbox],
                                outputs=[data_method_row, data_upload_row, data_generation_row, data_id_txtbox,
                                         data_success_row,
                                         success_msg, df, data_id, demo_title_dataset_loaded])

        # - Data Generation Column
        with data_generation_row:
            with gr.Column():
                generate_data_header = gr.Markdown("<h1 style='text-align: center;'> \u27A1 Generate Synthetic</h1>")
                synthetic_data_method = gr.Radio(choices=["Blobs", "Moons"],
                                                 label="Choose Synthetic Data Generation Type", value="Blobs")
                no_instances_input = gr.Number(label='No Instances', value=100)
                no_features_input = gr.Number(label='No Features', value=2)
                generate_data_btn = gr.Button('Generate Synthetic Data')
                generate_data_btn.click(generate_data, inputs=[synthetic_data_method, no_instances_input,
                                                               no_features_input, data_id_txtbox],
                                        outputs=[data_method_row, data_upload_row, data_generation_row,
                                                 data_success_row,
                                                 success_msg, df, data_id, demo_title_dataset_loaded])

    # <-----------------------------Exhaustive Search ✓-------------------------------------------------------------->
    with gr.Tab('Exhaustive Search'):
        default_search_space = """
                            {
                                "KMeans": {"n_clusters": [2, 21, 1], 
                                              "algorithm": ["lloyd", "elkan"],
                                              "max_iter": 500, 
                                              "init": ["k-means++", "random"]}, 
                                "DBSCAN": {"eps": [0.01, 1, 0.05], 
                                            "min_samples": [2, 21,1], 
                                            "metric": ["euclidean", "l1", "cosine"]}
                            }
                            """
        not_loaded_message_1 = gr.Markdown("""
                    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                        No data loaded yet.
                    </div>
                    """, elem_id="centered-message")

        # Section: 1 -> Perform Exhaustive Search

        with gr.Column(visible=False) as es_col:
            with gr.Column():
                gr.Markdown("<h1 style='text-align: center;'>Exhaustive Search</h1>")
                es_start_btn = gr.Button("Start Exhaustive Search")
            with gr.Row():
                with gr.Column():
                    search_space_input = gr.Textbox(value=default_search_space, label='Set Search Space',
                                                    interactive=True,
                                                    lines=12)
                with gr.Column():
                    index_radio = gr.Radio(["All", "Custom Set"], label="Select Indices", value="All")
                    index_dropdown = gr.Dropdown(choices=cvi_list, multiselect=True, label="Select Indexes",
                                                 visible=False,
                                                 interactive=True)
                    index_radio.change(
                        lambda x: gr.update(visible=True) if x == "Custom Set" else gr.update(visible=False),
                        inputs=index_radio, outputs=index_dropdown)

        with gr.Column(visible=False) as es_success_row:
            es_success_msg = gr.Markdown(visible=False)
            dl_results_btn = gr.DownloadButton("Download Results", "es_search_results.json", visible=False)

        # Section: 2 -> Exhaustive Search Results

        with gr.Column(visible=False) as es_results_row:
            gr.Markdown("<h1 style='text-align: center;'>Exhaustive Search Results</h1>")
            with gr.Row():
                hist_img = gr.Image(interactive=False)
                pie_img = gr.Image(interactive=False)

        es_start_btn.click(exhaustive_search, inputs=[master_results, data_id, df, search_space_input, index_radio,
                                                      index_dropdown], outputs=[master_results, es_success_msg,
                                                                                dl_results_btn, pie_img, hist_img,
                                                                                best_config_per_cvi])

    # <-----------------------------Clustering Exploration ✓--------------------------------------------------------->
    with gr.Tab('Clustering Exploration'):
        not_loaded_message_2 = gr.Markdown("""
                            <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                                No data loaded yet.
                            </div>
                            """, elem_id="centered-message")
        best_config_row = gr.Row(visible=False)
        results_explore_column = gr.Column(visible=False)

        with best_config_row:
            with gr.Column(scale=3):
                best_config_index_dropdown = gr.Dropdown(choices=cvi_list, multiselect=False,
                                                         label="Select Index", visible=True, interactive=True,
                                                         elem_id="my-dropdown")
            with gr.Column(scale=7):
                best_config = gr.Textbox(label="Best Configuration", elem_id="my-textbox")

        best_config_index_dropdown.change(return_best_cvi_config, inputs=[best_config_per_cvi,
                                                                          best_config_index_dropdown],
                                          outputs=[best_config_browse, best_config])

        with results_explore_column:
            gr.Markdown("Results Exploration")
            reduction_choices = gr.Radio(choices=["T-SNE", "PCA", "MDS"])
            reduction_msg = gr.Markdown(visible=False)
            with gr.Row():
                with gr.Column():
                    # Dim Reduction Functionality.

                    reduction_choices.change(dimensionality_reduction, inputs=[df, reduction_choices, df_reduced],
                                             outputs=[df_reduced, reduction_msg])
                    clusters_visualized = gr.Plot()
                    # Clustering Visualization

                with gr.Column():
                    heatmap_plot = gr.Plot()

            viz_btn = gr.Button("Visualize")

        viz_btn.click(serve_clustering_visualizations,
                      inputs=[best_config_per_cvi, df_reduced, best_config_index_dropdown, reduction_choices, df],
                      outputs=[clusters_visualized, heatmap_plot])

    # <-----------------------------Meta Learning --------------------------------------------------------->
    with gr.Tab('Meta-Learning'):
        mf_df = gr.State()
        not_loaded_message_3 = gr.Markdown("""
                            <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                                No data loaded yet.
                            </div>
                            """, elem_id="centered-message")

        with gr.Tab("Calculate Meta-Features"):
            generate_mf_col = gr.Column(visible=False)
        with gr.Tab("Train Meta-Learner"):
            train_ml_col = gr.Column(visible=False)
        with generate_mf_col:
            gr.Markdown("<h1 style='text-align: center;'> \u27A1 Generate Meta Features </h1>")
            with gr.Column():
                mf_calculate_btn = gr.Button(value="Calculate MF")
                add_mf_to_repo = gr.Button("Add to Repository", visible=False)
                download_mf_btn = gr.DownloadButton("json mf", visible=False)
            with gr.Column():
                mf_calculated = gr.JSON()

            mf_calculate_btn.click(calculate_mf, inputs=df, outputs=[mf_calculated, add_mf_to_repo, download_mf_btn])

        with train_ml_col:
            gr.Markdown("<h1 style='text-align: center;'> \u27A1 Train Meta Learner </h1>")
            ml_select = gr.Radio(["KNN", "DT"], label="Select Classifier")
            knn_options = {"no_neighbors": gr.Slider(2, 10, value=5, step=1, interactive=True,
                                                     label="Number of Neighbors", visible=False),
                           "metric": gr.Radio(choices=["euclidean", "l1", "cosine"], label='metric',
                                              visible=False, interactive=True)

                           }
            dt_options = {"criterion": gr.Radio(choices=["gini", "entropy", "log_loss"], label='metric',
                                                visible=False, interactive=True)
                          }
            ml_select.change(fn=update_ml_options, inputs=ml_select, outputs=[knn_options[key] for key in knn_options] +
                                                                             [dt_options[key] for key in dt_options])

            with gr.Row():
                with gr.Column(elem_id="fixed_height_col"):
                    gr.Markdown("Target Variable")
                    best_alg_selection = gr.Radio(["Most Popular Alg", "Specific CVI"])
                    best_config_ml = gr.Dropdown(choices=cvi_list, multiselect=False,
                                                 label="Select Index", visible=False, interactive=True)
                    best_alg_selection.change(
                        lambda x: gr.update(visible=True) if x == "Specific CVI" else gr.update(visible=False),
                        inputs=best_alg_selection, outputs=best_config_ml)

                # Meta Features Selection
                with gr.Column():
                    gr.Markdown("<h2>Select Meta-Features</h2>")
                    mf_selection = gr.Radio(["All",  "Custom Selection"])

                    mf_search_type = gr.Radio(["Category", "Paper"], label="Search Type", visible=False)
                    mf_search_choices = gr.Dropdown(choices=[], multiselect=True, interactive=True, visible=False)
                    mf_search_type.change(update_search_type_dropdown, inputs=mf_search_type, outputs=mf_search_choices)

                    mf_selection.change(fn=toggle_mf_selection_options, inputs=[mf_selection], outputs=[mf_search_type, mf_search_choices])




    df.change(enable_tabs_after_df, outputs=[es_col, es_success_row, es_results_row, not_loaded_message_1,
                                             best_config_row, results_explore_column, not_loaded_message_2,
                                             generate_mf_col, train_ml_col, not_loaded_message_3])

demo.launch()
