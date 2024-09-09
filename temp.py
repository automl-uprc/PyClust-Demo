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







                generate_cluster_labels_btn = gr.Button('Generate Cluster Labels')
                generate_cluster_labels_btn.click(generate_clusters, inputs=temp_slider_kmeans,
                                        outputs=[labels_row, labels_success_row, success_msg_labels, labels])



    with gr.Tab('CVI Calculation'):
        with gr.Row() as single_cvi_row:
            with gr.Column():
                cvi_select = gr.Textbox(label="Select CVI:")
                calculate_single_cvi_btn = gr.Button("Calculate CVI")
                cvi_calc_success_msg = gr.Markdown(visible=False)
                execution_plan = gr.Image(interactive=False, label='Execution Plan')
                subprocess_dropdown = gr.Dropdown(choices=[x for x in common_operations], label="Visualize Subprocess")
                subprocess_dropdown.change(visualize_subgraph_as_tree, inputs=subprocess_dropdown, outputs=execution_plan)
                calculate_single_cvi_btn.click(fn=calculate_cvi, inputs=[df, labels, cvi_select],
                                               outputs=[cvi_calc_success_msg])

        with gr.Row() as all_cvi_row:
            with gr.Column():
                calculate_btn = gr.Button("Calculate all CVI")
                all_cvi_json = gr.Json()
                calculate_btn.click(calculate_cvi, inputs=[df, labels], outputs=[all_cvi_json])
demo.launch()



