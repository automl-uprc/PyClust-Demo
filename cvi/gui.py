import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from sklearn.cluster import *
from autometrics import AutoMetrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def display_visualizations(file, model_choice, n_clusters, affinity, linkage, eps, min_samples, metric, tol, algorithm,
                           init, damping, gamma, n_neighbors, assign_labels, cluster_method, xi, max_eps, threshold):
    # Read the uploaded CSV file
    df = pd.read_csv(file.name)

    # Generate visualizations
    visualizations = []

    hps = {'n_clusters': n_clusters, 'affinity': affinity, 'linkage': linkage, 'eps': eps,
           'min_samples': min_samples, 'metric': metric, 'tol': tol, 'algorithm': algorithm,
           'init': init, 'damping': damping, 'gamma': gamma, 'n_neighbors': n_neighbors,
           'assign_labels': assign_labels, 'cluster_method': cluster_method, 'xi': xi,
           'max_eps': max_eps, 'threshold': threshold}

    ap = AgglomerativeClustering(n_clusters=hps['n_clusters'], metric=hps['metric'], linkage=hps['linkage']).fit(df)

    #vis = pd.DataFrame(model_eval(df, model_choice, hps)).plot.bar().get_figure()

    plt.figure(figsize=(8, 8))
    vis = sns.barplot({'silhouette': silhouette_score(df, ap.labels_), 'ch': calinski_harabasz_score(df, ap.labels_)}, orient='x')
    vis_fig = vis.get_figure()
    vis_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    vis_fig.savefig(vis_file.name)
    plt.close(vis_fig)


    return vis_file.name


def hyperparameters_selection(model):
    # Visibility: n_clusters, affinity, linkage, eps, min_samples, metric, tol, algorithm, init, damping, gamma, n_neighbors, assign_labels, cluster_method, xi, max_eps, threshold
    if model == 'Agglomerative Clustering':
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False)
    elif model == 'DBSCAN':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False)
    elif model == 'KMeans':
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(
            visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False)
    elif model == 'AffinityPropagation':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False)
    elif model == 'SpectralClustering':
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(
            visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False)
    elif model == 'OPTICS':
        return gr.update(visivle=False), gr.update(visivle=False), gr.update(visivle=False), gr.update(
            visivle=False), gr.update(visivle=True), gr.update(visivle=True), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=True), gr.update(visible=False)
    elif model == 'Birch':
        return gr.update(visivle=False), gr.update(visivle=False), gr.update(visivle=False), gr.update(
            visivle=False), gr.update(visivle=False), gr.update(visivle=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=True)


def model_eval(file, model_choice, hps):
    if model_choice == 'Agglomerative Clustering':
        model = AgglomerativeClustering(n_clusters=hps['n_clusters'], affinity=hps['affinity'], linkage=hps['linkage'])
    elif model_choice == 'DBSCAN':
        model = DBSCAN(eps=hps['eps'], min_samples=hps['min_samples'], metric=hps['metric'], n_jobs=-1)
    elif model_choice == 'KMeans':
        model = KMeans(n_clusters=hps['n_clusters'], tol=hps['tol'], algorithm=hps['algorithm'], max_iter=500,
                       init=hps['init'])
    elif model_choice == 'Affinity Propagation':
        model = AffinityPropagation(damping=hps['damping'], max_iter=500)
    elif model_choice == 'MeanShift':
        model = MeanShift(n_jobs=-1, max_iter=500)
    elif model_choice == 'Spectral Clustering':
        model = SpectralClustering(n_clusters=hps['n_clusters'], affinity=hps['affinity'], gamma=hps['gamma'],
                                   n_neighbors=hps['n_neighbors'], assign_labels=hps['assign_labels'], n_jobs=-1)
    elif model_choice == 'OPTICS':
        model = OPTICS(min_samples=hps['min_samples'], metric=hps['metric'], cluster_method=hps['cluster_method'],
                       xi=hps['xi'], n_jobs=-1, max_eps=hps['max_eps'])
    elif model_choice == 'Birch':
        model = Birch(threshold=hps['threshold'], n_clusters=None)

    am = AutoMetrics(file, model.fit(file).labels_)
    cvis = am.cvis

    return cvis


def build_interface():
    with gr.Blocks() as interface:
        with gr.Column():
            file_input = gr.File(label="Upload CSV")

            model_choice = gr.Radio(choices=['Agglomerative Clustering', 'DBSCAN', 'KMeans', 'Affinity Propagation',
                                             'MeanShift', 'Spectral Clustering', 'OPTICS', 'Birch',
                                             'Mini-Batch KMeans'], label='Model Selection')

            # Hyperparameters
            n_clusters = gr.Slider(2, 20, 1, label='n_clusters', visible=False)
            affinity = gr.Radio(choices=['euclidean', 'l1', 'cosine'], label='affinity', visible=False)
            linkage = gr.Radio(choices=['ward', 'average', 'complete', 'single'], label='linkage', visible=False)
            eps = gr.Slider(0.0001, 1, 50, label='eps', visible=False)
            min_samples = gr.Slider(2, 20, 1, label='min_samples', visible=False)
            metric = gr.Radio(choices=['euclidean', 'l1', 'cosine'], label='metric', visible=False)
            tol = gr.Slider(0.0001, 0.0002, 10, label='tol', visible=False)
            algorithm = gr.Radio(choices=['lloyd', 'elkan'], label='algorithm', visible=False)
            init = gr.Radio(choices=['k-means++', 'random'], label='init', visible=False)
            damping = gr.Slider(0.1, 1, 10, label='damping', visible=False)
            gamma = gr.Slider(1, 2, 0.1, label='gamma', visible=False)
            n_neighbors = gr.Slider(8, 10, 1, label='n_neighbors', visible=False)
            assign_labels = gr.Radio(choices=['kmeans', 'discretize', 'cluster_qr'], label='assign_labels',
                                     visible=False)
            cluster_method = gr.Radio(choices=['xi', 'dbscan'], label='cluster_method', visible=False)
            xi = gr.Slider(0.03, 0.15, 0.5, label='xi', visible=False)
            max_eps = gr.Slider(0.0001, 1, 20, label='max_eps', visible=False)
            threshold = gr.Slider(0.0001, 1, 100, label='threshold', visible=False)

            model_choice.change(hyperparameters_selection, inputs=model_choice,
                                outputs=[n_clusters, affinity, linkage, eps, min_samples, metric, tol,
                                         algorithm, init, damping, gamma, n_neighbors, assign_labels,
                                         cluster_method, xi, max_eps, threshold])

            #cvis = gr.Image(label="CVIs")

            button = gr.Button("Evaluate Clustering")

            print(model_choice)
            button.click(display_visualizations, inputs=[file_input, model_choice, n_clusters, affinity, linkage, eps,
                                                         min_samples, metric, tol, algorithm, init, damping, gamma,
                                                         n_neighbors, assign_labels, cluster_method, xi, max_eps,
                                                         threshold],
                         outputs=gr.Image(label="CVIs"))

    return interface


if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
