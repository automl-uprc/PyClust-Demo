import gradio as gr

from pyclustkit.metalearning import  MFExtractor
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA

mfe = MFExtractor()
mf_categories = mfe.mf_categories
mf_papers = mfe.mf_papers
all_mf = [key for key in mfe.meta_features]






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