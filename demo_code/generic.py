import gradio as gr


def on_df_load(df):
    """
    Updates on UI/cache after dataset is provided.
    Returns:
        (tuple):
            (0) df_features_more_than_two: Bool
            (1) Grid Search Tab: Visibility
            (2) Clustering Exploration Tab: Visibility
            (3) Not Loaded Message: Visibility
    """
    if df.shape[1] > 2:
        needs_reduction = True
    else:
        needs_reduction = False

    return needs_reduction, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

def df_needs_dimred(df_needs_dim_red):
    if df_needs_dim_red:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

