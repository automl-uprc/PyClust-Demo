from tab_code.tab_1_code import *
from tab_code.tab_2_code import *
from tab_code.tab_3_code import (
                                 return_best_cvi_config,  statistics_per_cluster, select_option,
                                 dimensionality_reduction, update_ui)
from tab_code.tab_4_code import update_search_type_dropdown, calculate_mf, display_df
from css import custom_css


def update_cache_results(data_id_, master_results_):
    master_results_[data_id_] = {"KMeans": [], "DBSCAN": []}
    return master_results_


# Create the Gradio interface
with gr.Blocks(css=custom_css) as demo:
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
    demo_title = gr.Markdown("<h1 style='text-align: center; overflow: hidden;'>PyClust Demo</h1>")
    demo_title_dataset_loaded = gr.Markdown("<h2 style='text-align: center;'>No dataset loaded.</h2>")

    # <-----------------------------Data Loading ✓-------------------------------------------------------------->
    with gr.Tab('Data Loading/Generation'):
        data_id_txtbox = gr.Textbox(label="Assign dataset ID")

        # Content
        data_upload_row = gr.Row()
        data_generation_row = gr.Row()

        # Only visible after dataset is loaded or generated.
        with gr.Column(visible=False) as data_success_row:
            success_msg = gr.Markdown()
            change_data_btn = gr.Button("Change Dataset?")
            change_data_btn.click(change_data_update_visibility, outputs=[data_upload_row, data_generation_row,
                                                                          data_id_txtbox, data_success_row])

        # - Data Upload Column
        with data_upload_row:
            with gr.Column():
                load_data_header = gr.Markdown(
                    "<h1 style='text-align: center;'> \u27A1 Load Data ([.csv])</h1>")
                upload_csv_options = gr.CheckboxGroup(["Headers", "Scale Data (MinMax)"],
                                                      label="Preprocessing Options")
                csv_file = gr.File(label="Upload your CSV file")
                csv_file.change(load_csv, inputs=[csv_file, upload_csv_options, data_id_txtbox],
                                outputs=[data_upload_row, data_generation_row, data_id_txtbox, data_success_row,
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
                                        outputs=[data_upload_row, data_generation_row, data_id_txtbox, data_success_row,
                                                 success_msg, df, data_id, demo_title_dataset_loaded])

    # <-----------------------------Exhaustive Search ✓--------------------------------------------------------->
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
        gr.Markdown("<h1 style='text-align: center;'>Exhaustive Search</h1>")

        # Section: 1 -> Perform Exhaustive Search
        es_start_btn = gr.Button("Start Exhaustive Search")
        with gr.Row():
            with gr.Column():
                search_space_input = gr.Textbox(value=default_search_space, label='Set Search Space', interactive=True,
                                                lines=12)
            with gr.Column():
                index_radio = gr.Radio(["All", "Custom Set"], label="Select Indices", value="All")
                index_dropdown = gr.Dropdown(choices=cvi_list, multiselect=True, label="Select Indexes", visible=False,
                                             interactive=True)
                index_radio.change(lambda x: gr.update(visible=True) if x == "Custom Set" else gr.update(visible=False),
                                   inputs=index_radio, outputs=index_dropdown)

        es_success_msg = gr.Markdown(visible=False)
        dl_results_btn = gr.DownloadButton("Download Results", "es_search_results.json", visible=False)

        # Section: 2 -> Exhaustive Search Results
        gr.Markdown("<h1 style='text-align: center;'>Exhaustive Search Results</h1>")
        with gr.Row():
            with gr.Column():
                hist_img = gr.Image(interactive=False)
            with gr.Column():
                pie_img = gr.Image(interactive=False)

        es_start_btn.click(exhaustive_search, inputs=[master_results, data_id, df, search_space_input, index_radio,
                                                      index_dropdown], outputs=[master_results, es_success_msg,
                                                                                dl_results_btn, pie_img, hist_img,
                                                                                best_config_per_cvi])

    # <-----------------------------Clustering Exploration --------------------------------------------------------->
    with gr.Tab('Clustering Exploration'):
        gr.Markdown("Find configuration that optimizes index")
        with gr.Row():
            with gr.Column(scale=2):
                best_config_index_dropdown = gr.Dropdown(choices=cvi_list, multiselect=False,
                                                         label="Select Index", visible=True, interactive=True)
            with gr.Column(scale=7):
                best_config = gr.Textbox(label="Best Configuration")

            with gr.Column(scale=1):
                select_this_best = gr.Checkbox(label="Select")

        best_config_index_dropdown.change(return_best_cvi_config, inputs=[best_config_per_cvi,
                                                                          best_config_index_dropdown],
                                          outputs=[best_config_browse, best_config])

        gr.Markdown("Or select custom configuration/ Apply if it does not exist")

        with gr.Row():
            with gr.Column(scale=7):
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
                dbscan_options = {"eps": gr.Slider(0.01, 1, value=0.1, step=0.05, interactive=True,
                                                   label="eps", visible=False),
                                  "min_samples": gr.Slider(2, 21, value=2, step=1, interactive=True,
                                                           label="min samples", visible=False),
                                  "metric": gr.Radio(choices=["euclidean", "l1", "cosine"], label='metric',
                                                     visible=False, interactive=True)
                                  }
                all_options = [x for x in list(kmeans_options.values()) + list(dbscan_options.values())]

            with gr.Column(scale=1):
                select_this_best_2 = gr.Checkbox(label="select", interactive=False)

            # Update ui based on algorithm selected.
            algorithm_select_dropdown.change(fn=update_           ui, inputs=algorithm_select_dropdown,
                                             outputs=[kmeans_options[key] for key in kmeans_options] +
                                                     [dbscan_options[key] for key in dbscan_options])

        # On change function to ensure only one or neither option is selected.
        select_this_best.select(select_option, inputs=[select_this_best, select_this_best_2, best_config_browse],
                                outputs=[labels, select_this_best_2, select_this_best])
        select_this_best_2.select(select_option, inputs=[select_this_best, select_this_best_2, best_config_browse],
                                  outputs=[labels, select_this_best, select_this_best_2])

        check_config_exists_btn = gr.Button("Check if Configuration Exists")

        # The following appear after the user checks if the configuration selected exists in the repository
        explore_msg_1 = gr.Markdown("Configuration not Found ! Press the button below to apply clustering",
                                    visible=False)
        apply_clustering_btn = gr.Button("Apply Configuration", visible=False)

        explore_msg_2 = gr.Markdown("Configuration Found ! Select it to explore further", visible=False)

        check_config_exists_btn.click(check_if_config_exists, inputs=[data_id, master_results,
                                                                      algorithm_select_dropdown] +
                                                                     [kmeans_options[key] for key in kmeans_options] +
                                                                     [dbscan_options[key] for key in dbscan_options],
                                      outputs=[explore_msg_1, apply_clustering_btn, explore_msg_2, select_this_best_2])

        with gr.Row():
            with gr.Column():
                # Dim Reduction Functionality.
                reduction_choices = gr.Radio(choices=["T-SNE", "PCA", "MDS"])
                reduction_msg = gr.Markdown(visible=False)
                reduction_choices.change(dimensionality_reduction, inputs=[df, reduction_choices, df_reduced],
                                         outputs=[df_reduced, reduction_msg])

                # Clustering Visualization
                clusters_visualized = gr.Image(interactive=False)
                viz_btn = gr.Button("Visualize")
                viz_btn.click( inputs=[master_results, df, data_id, df_reduced, reduction_choices,
                                                                      select_this_best, select_this_best_2, best_config,
                                                                      algorithm_select_dropdown] +
                                                                     [kmeans_options[key] for key in kmeans_options] +
                                                                     [dbscan_options[key] for key in dbscan_options],
                              outputs=clusters_visualized
                              )

            with gr.Column():
                statistics_dataframe = gr.Dataframe()
                stats_per_cluster_btn = gr.Button("Get Per Cluster Statistics")
                stats_per_cluster_btn.click(statistics_per_cluster, inputs=[df, labels], outputs=statistics_dataframe)

    with gr.Tab('Meta-Learning'):
        mf_df = gr.State()
        with gr.Row():
            with gr.Column():
                gr.Markdown("Select Meta-Features to Generate")
                mf_search_type = gr.Radio(["Category", "Paper", "Custom Selection"], label="Search Type")
                mf_search_choices = gr.Dropdown(choices=[], multiselect=True, interactive=True)
                mf_search_type.change(update_search_type_dropdown, inputs=mf_search_type, outputs=mf_search_choices)
                mf_calculate_btn = gr.Button(value="Calculate MF")

                gr.Markdown("Generate Meta-Record")
                add_mf_to_repo = gr.Button("Add to Repository")


            with gr.Column():
                mf_calculated = gr.JSON()

            mf_calculate_btn.click(calculate_mf, inputs=[df, mf_search_type, mf_search_choices],
                                   outputs=[mf_calculated])

        with gr.Row():
            best_config_ml = gr.Dropdown(choices=cvi_list, multiselect=False,
                                         label="Select Index", visible=True, interactive=True)


        with gr.Row():
            with gr.Column():
                gr.Markdown("Train Meta Learner")
                gr.Radio(["KNN", "DT"])
                gr.Dropdown("no_neighbors")
                gr.Dataframe()

    with gr.Tab("Repository"):
        clustering_parameter_records = gr.Dataframe()

        algorithm_results_select_dropdown = gr.Dropdown(["KMeans", "DBSCAN"],
                                                        label='Select Clustering Algorithm')
        algorithm_results_select_dropdown.change(display_df,
                                                 inputs=[algorithm_results_select_dropdown, master_results],
                                                 outputs=clustering_parameter_records)
demo.launch()
