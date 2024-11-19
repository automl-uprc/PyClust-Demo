import gradio as gr

from pyclustkit.metalearning import  MFExtractor
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA

mfe = MFExtractor()
mf_categories = mfe.mf_categories
mf_papers = mfe.mf_papers
all_mf = [key for key in mfe.meta_features]


def dimensionality_reduction(df, method, df_reduced):
    print(df)
    print(method)
    print(df_reduced)
    methods = {"T-SNE": TSNE, "PCA": PCA, "MDS": MDS}
    if df_reduced[method] is None:
        df_reduced[method] = methods[method](n_components=2).fit_transform(df)
        return df_reduced, gr.update(visible=True,
                                     value="First Trying Applying This Method, it may take a while for big "
                                           "datasets")
    else:
        return df_reduced, gr.update(visible=False)


def return_best_cvi_config(best_config_per_cvi, cvi):
    """
    Returns the best configuration according to the results' dict.
    Args:
        best_config_per_cvi (dict):
        cvi (str):

    Returns:

    """
    return (best_config_per_cvi[cvi],
            f"Algorithm: {best_config_per_cvi[cvi]['algorithm']}\nParameters: {best_config_per_cvi[cvi]['params']}")

def update_ml_options(selected_ml):
    """
    Visibility of meta-lerner classifier parameters
    Args:
        selected_ml (str):

    Returns:

    """
    if selected_ml == "KNN":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    elif selected_ml == "DT":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def update_search_type_dropdown(method):
    var = {"Category": mf_categories,
           "Paper": mf_papers,
           "Custom Selection": all_mf}
    return gr.update(value="", choices=var[method])


def toggle_mf_selection_options(method):
    if method == "Custom Selection":
        return gr.update(visible=True), gr.update(visible=True)
    elif method == "All":
        return gr.update(visible=False), gr.update(visible=False)