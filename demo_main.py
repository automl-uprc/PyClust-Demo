import gradio as gr
from tab_1_code import load_csv, generate_data
from tab_2_code import update_ui, apply_clustering, display_df
from tab_3_code import cvi_list, exhaustive_search, find_best, find_best_per_cvi, return_best_cvi_config
from css import custom_css

print(gr.__version__)

theme = gr.themes.Base(
    primary_hue="stone",
    secondary_hue="indigo",
    text_size="lg",
    spacing_size="sm",
).set(
    button_primary_background_fill="#FF0000",
    button_primary_background_fill_dark="#AAAAAA"
)

# Create the Gradio interface
with gr.Blocks(theme=theme, css=custom_css) as demo:
    df = gr.State()
    labels = gr.State()
    master_results = gr.State({"KMeans": [], "DBSCAN": []})
    best_config_per_cvi = gr.State({})





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
                                                 label="Choose Synthetic Data Generation Type", value="Blobs")
                no_instances_input = gr.Number(label='No Instances', value=100)
                no_features_input = gr.Number(label='No Features', value=2)
                generate_data_btn = gr.Button('Generate Synthetic Data')
                generate_data_btn.click(generate_data, inputs=[synthetic_data_method, no_instances_input,
                                                               no_features_input],
                                        outputs=[data_upload_row, data_generation_row,
                                                 load_data_header, generate_data_header,
                                                 data_success_row, success_msg, df])
    with gr.Tab('Clustering'):
        with gr.Row():
            with gr.Column():
                gr.Markdown("<h1 style='text-align: center;'>Apply a Single Clustering Algorithm</h1>")








    with gr.Tab('Exhaustive Search'):
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

        es_start_btn = gr.Button("Start Exhaustive Search")
        with gr.Row():
            with gr.Column():
                search_space_input = gr.Textbox(value=default_search_space,
                                                label='Set Search Space', interactive=True, lines=12)
            with gr.Column():
                index_radio = gr.Radio(["All", "Custom Set"], label="Select Indices", value="All")
                index_dropdown = gr.Dropdown(choices=cvi_list, multiselect=True, label="Select Indexes", visible=False,
                                             interactive=True)
                index_radio.change(lambda x: gr.update(visible=True) if x == "Custom Set" else gr.update(visible=False),
                                   inputs=index_radio, outputs=index_dropdown)

        dl_results_btn = gr.DownloadButton("Download Results", "es_search_results.json", visible=False)
        # dl_results_btn.click(create_json, inputs=master_results, outputs=gr.File())

        es_success_msg = gr.Markdown(visible=False)

        gr.Markdown("<h1 style='text-align: center;'>Exhaustive Search Results</h1>")
        with gr.Row():
            with gr.Column():
                hist_img = gr.Image(interactive=False)
            with gr.Column():
                pie_img = gr.Image(interactive=False)

        es_start_btn.click(exhaustive_search, inputs=[master_results, df, search_space_input, index_radio,
                                                      index_dropdown], outputs=[master_results, es_success_msg,
                                                                                dl_results_btn, pie_img, hist_img,
                                                                                best_config_per_cvi])
    with gr.Tab('Clustering Exploration'):
        gr.Markdown("Find configuration that optimizes index")
        with gr.Row():
            with gr.Column():
                best_config_index_dropdown = gr.Dropdown(choices=cvi_list, multiselect=False,
                                                         label="Select Index", visible=True, interactive=True)
            with gr.Column():
                best_config = gr.Textbox(label="Best Configuration")

            with gr.Column():
                select_this_best = gr.Checkbox()

        best_config_index_dropdown.change(return_best_cvi_config, inputs=[best_config_per_cvi,
                                                                          best_config_index_dropdown],
                                          outputs=best_config)

        gr.Markdown("Or select custom configuration/ Apply if it does not exist")

        with gr.Row():
            with gr.Column():
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

            with gr.Column():
                select_this_best_2 = gr.Checkbox()

            # Update ui based on algorithm selected.
            algorithm_select_dropdown.change(fn=update_ui, inputs=algorithm_select_dropdown,
                                             outputs=[kmeans_options[key] for key in kmeans_options] +
                                                     [dbscan_options[key] for key in dbscan_options])

        check_config_exists_btn = gr.Button("Check if Configuration Exists")
        apply_clustering_btn = gr.Button("Apply Configuration", visible=False)



        with gr.Row():
            with gr.Column():
                gr.Image()

            with gr.Column():
                gr.Dataframe()



    with gr.Tab("Repository"):
        clustering_parameter_records = gr.Dataframe()


        algorithm_results_select_dropdown = gr.Dropdown(["KMeans", "DBSCAN"],
                                                                label='Select Clustering Algorithm')
        algorithm_results_select_dropdown.change(display_df,
                                                         inputs=[algorithm_results_select_dropdown, master_results],
                                                         outputs=clustering_parameter_records)
demo.launch()
