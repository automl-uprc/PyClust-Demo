import gradio


from demo_code.tab_1 import *
from demo_code.tab_2 import *
from demo_code.tab_3 import *




import os


logo_path = 'clustering_framework_logo.jpg'
css = """
/* Customize tab label font size */
.tabs button {
    font-size: 18px;  /* Adjust font size as desired */
    font-weight: bold;  /* Optional: make text bold */}
    

/* Style the span inside the button */
#my_accordion button span {
    font-weight: bold; /* Bold text for emphasis */
    font-size: 20px; /* Even larger text for span */
}

#border_col {
    border: 2px solid #853403;
    padding: 10px;
    margin: 5px;
    border-radius: 5px;
}
"""

def update_cache_results(data_id_, master_results_):
    master_results_[data_id_] = {"KMeans": [], "DBSCAN": []}
    return master_results_


def control_data_visibility(data_method):
    """
    Changes layout for data upload/generation section
    Args:
        data_method (str): will be either Upload or Generate

    Returns:
        - 1 visibility update: Data upload Column
        - 2 visibility update: Data Generation Column
    """

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
    with gr.Row(visible=False) as title_row:
        demo_mf_completed = gr.Markdown("<h2 style='text-align: right; color:#3ebefe;'>Meta-Features: ❌</h2>")
        demo_ms_completed = gr.Markdown("<h2 style='text-align: left; color:#3ebefe;'>Model Search: ❌</h2>")


    # <-----------------------------Data Loading ✓-------------------------------------------------------------->
    with gr.Tab('(1) Data Loading/Generation'):


        with gr.Accordion("Usage Manual", open=True, elem_id="my_accordion"):
            gr.Markdown("""
                        # Welcome to the **PyClust Demo**! 🎉

                        ## Get Started:
                        On this page, you can:
                        1. **Load or Generate Data** 📊  
                        2. **Compute Meta-Features** 🔍  
                        3. **Select the Best Algorithm** for your dataset 🧠  
                        
                        ## What's Next?
                        - Dive into **Model Search** to train and evaluate different models.  
                        - Explore the **Repository** for pre-trained model selection meta-learners or train  and train new.. 

                        """)

        with gr.Accordion("Load Data", open=False, elem_id="my_accordion"):
            # Only visible after dataset is loaded or generated.
            with gr.Column(visible=False) as data_success_row:
                success_msg = gr.Markdown()
                change_data_btn = gr.Button("Change Dataset")

            with gr.Row() as data_method_row:
                data_id_textbox = gr.Textbox(label="Assign dataset ID", elem_classes="center-label")
                data_method = gr.Radio(choices=["Upload", "Generate"], label="Select How to Provide Data",
                                       elem_classes="center-radio")

                # - Data Upload Column
            with gr.Column(visible=False) as data_upload_row:

                load_data_header = gr.Markdown(
                            "<h1 style='text-align: center;'> \u27A1 Load Data ([.csv])</h1>")
                upload_csv_options = gr.CheckboxGroup(["Headers", "Scale Data (MinMax)"],
                                                              label="Preprocessing Options")
                csv_file = gr.File(label="Upload your CSV file")


                # - Data Generation Column
            with gr.Column(visible=False) as data_generation_row:
                generate_data_header = gr.Markdown(
                            "<h1 style='text-align: center;'> \u27A1 Generate Synthetic</h1>")
                synthetic_data_method = gr.Radio(choices=["Blobs", "Moons"],
                                                         label="Choose Synthetic Data Generation Type", value="Blobs")
                no_instances_input = gr.Number(label='No Instances', value=100)
                no_features_input = gr.Number(label='No Features', value=2)
                generate_data_btn = gr.Button('Generate Synthetic Data')

        with gr.Accordion("Calculate Meta Features", open=False, elem_id="my_accordion"):
            mf_calculate_btn = gr.Button(value="Calculate MF")
            download_mf_btn = gr.DownloadButton("Download (JSON)", visible=False)
            mf_calculated = gr.JSON()

        with gr.Accordion("Algorithm Selection!", open=False, elem_id="my_accordion"):
            gr.Markdown("""Prediction : KMEANS""", visible=False)
            with gr.Column():
                gr.Dropdown(["select Meta Learner", "ML model 2"], interactive=True)
                gr.Button("Predict !")

        # On change for tab 1
        csv_file.change(load_csv, inputs=[csv_file, upload_csv_options, data_id_textbox],
                            outputs=[data_method_row, data_upload_row, data_generation_row, data_id_textbox,
                                     data_success_row, success_msg, df, data_id, demo_title_dataset_loaded, title_row])
        generate_data_btn.click(generate_data, inputs=[synthetic_data_method, no_instances_input,
                                                                       no_features_input, data_id_textbox],
                                                outputs=[data_method_row, data_upload_row, data_generation_row,
                                                         data_success_row, success_msg, df, data_id,
                                                         demo_title_dataset_loaded, title_row])

        change_data_btn.click(change_data_update_visibility, outputs=[data_method_row, data_success_row, data_method])



        data_method.change(control_data_visibility, inputs=data_method, outputs=[data_generation_row,
                                                                                      data_upload_row])

        mf_calculate_btn.click(mf_process, inputs=[df, data_id], outputs= [mf_calculated,download_mf_btn,
                                                                           demo_mf_completed]) #

    # <-----------------------------Exhaustive Search ✓-------------------------------------------------------------->
    with gr.Tab('(2) Parameter Search'):
        with gr.Tab("Grid Search"):
            default_search_space = """
                                 {
                                    "KMeans": {"n_clusters": [2, 21, 1], 
                                                  "algorithm": ["lloyd", "elkan"],
                                                  "max_iter": 500, 
                                                  "init": ["k-means++", "random"]}, 
                                                  
                                    "DBSCAN": {"eps": [0.01, 1, 0.05], 
                                                "min_samples": [2, 21,1], 
                                                "metric": ["euclidean", "l1", "cosine"]}, 
                                                
                                    "Agglomerative": {"n_clusters": [2, 21, 1], 
                                                "affinity": ["euclidean", "l1", "cosine"], 
                                                "linkage": ["ward", "complete", "average", "single"]}, 
                                    
                                    "Affinity Propagation": {"damping": [0.1, 1, 0.1]}, 
                                    
                                    "Spectral Clustering": {"n_clusters": [2, 21, 1], 
                                                  "gamma": [1.0, 2.0, 0.1],
                                                  "affinity": ["nearest_neighbors", "rbf"], 
                                                  "n_neighbors": [2, 10, 1],
                                                  "assign_labels": ["kmeans", "discretize", "cluster_qr"]}   
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
                        search_space_input = gr.Textbox(value=default_search_space, label='Set Search Space',
                                                        interactive=True,
                                                        lines=12)


            with gr.Column(visible=False) as es_success_row:
                es_success_msg = gr.Markdown(visible=False)
                dl_results_btn = gr.DownloadButton("Download Results", "es_search_results.json", visible=False)

            # Section: 2 -> Exhaustive Search Results

            with gr.Column(visible=False) as es_results_row:
                gr.Markdown("<h1 style='text-align: center;'>Exhaustive Search Results</h1>")
                with gr.Row():
                    hist_img = gr.Image(interactive=False)
                    pie_img = gr.Image(interactive=False)

            es_start_btn.click(exhaustive_search, inputs=[master_results, data_id, df, search_space_input],
                               outputs=[master_results, es_success_msg, dl_results_btn, pie_img, hist_img,
                                        best_config_per_cvi, demo_ms_completed])


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

    # <-----------------------------Repository --------------------------------------------------------->
    with gr.Tab('(3) Repository'):
        gr.Markdown("<h2 style='text-align: center; color:#3ebefe;'>Number of Datasets in The Repository: </h2>")

        with gr.Accordion("Trained Meta-Learners", elem_id="my_accordion"):
            gr.Markdown("""
            A list of all the trained meta-learners along with their meta-data. 
            """)
            gr.Dataframe(pd.DataFrame(columns=["Meta-Learner", "Accuracy", "Datasets Used", "Parameters", "Training Date"]))

        with gr.Accordion("Train Meta Learner", elem_id="my_accordion"):
            gr.Markdown("""To train a new meta-learner please configure: (a) which classifier to user (meta-learner)
                            , (b) which meta-features to use for the datasets (independent variables), (c) how to select the best
                             algorithm for each dataset (dependent variable) """)
            with gr.Row():

                with gr.Column(elem_id="border_col"):
                    gr.Markdown("""(A) Configure The Meta Learner Algorithm""")
                    ml_select = gr.Radio(["KNN", "DT"], label="Select Classifier")
                    knn_options = {"no_neighbors": gr.Slider(2, 10, value=5, step=1, interactive=True,
                                                             label="Number of Neighbors", visible=False),
                                   "metric": gr.Radio(choices=["euclidean", "l1", "cosine"], label='metric',
                                                      visible=False, interactive=True)

                                   }
                    dt_options = {"criterion": gr.Radio(choices=["gini", "entropy", "log_loss"], label='metric',
                                                        visible=False, interactive=True)
                                  }
                    ml_select.change(fn=update_ml_options, inputs=ml_select,
                                     outputs=[knn_options[key] for key in knn_options] +
                                             [dt_options[key] for key in dt_options])

                with gr.Column(elem_id="border_col"):
                    gr.Markdown("""(B) Select The meta-features group to include as training variables""")
                    mf_selection = gr.Radio(["All", "Custom Selection"])

                    mf_search_type = gr.Radio(["Category", "Paper"], label="Search Type", visible=False)
                    mf_search_choices = gr.Dropdown(choices=[], multiselect=True, interactive=True, visible=False)
                    mf_search_type.change(update_search_type_dropdown, inputs=mf_search_type,
                                          outputs=mf_search_choices)

                    mf_selection.change(fn=toggle_mf_selection_options, inputs=[mf_selection],
                                        outputs=[mf_search_type, mf_search_choices])



                with gr.Column(elem_id="border_col"):
                    gr.Markdown("""(C) Select how to appoint the best algorithm to each dataset""")
                    best_alg_selection = gr.Radio(["Most Popular Alg", "Specific CVI"])
                    best_config_ml = gr.Dropdown(choices=cvi_list, multiselect=False,
                                                 label="Select Index", visible=False, interactive=True)
                    best_alg_selection.change(
                        lambda x: gr.update(visible=True) if x == "Specific CVI" else gr.update(visible=False),
                        inputs=best_alg_selection, outputs=best_config_ml)

    df.change(enable_tabs_after_df, outputs=[es_col, es_success_row, es_results_row, not_loaded_message_1,
                                             best_config_row, results_explore_column, not_loaded_message_2,
                                             ])



if __name__ == "__main__":
    # Create repo folders for meta-features and es results
    cwd = os.getcwd()
    if not os.path.isdir(os.path.join(cwd, "results")):
        os.mkdir(os.path.join(cwd, "results"))

    mf_path = os.path.join(cwd, "results", "mf")
    es_path = os.path.join(cwd, "results", "es")

    if not os.path.isdir(mf_path):
        os.mkdir(mf_path)

    if not os.path.isdir(es_path):
        os.mkdir(es_path)



    demo.launch(share=False)


