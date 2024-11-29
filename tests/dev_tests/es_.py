from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, AffinityPropagation, SpectralClustering
from itertools import product
from demo_code.tab_2 import transform_search_space
import json
from sklearn.datasets import make_blobs

from main import best_config_per_cvi

x,y = make_blobs(n_samples=100, n_features=2)
# (1) JSON with search space is loaded correctly
space = """
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

j_space = json.loads(space)
print(j_space.keys())
transform_search_space(j_space)

# (2) Exhaustive Search -------------------------------------------------------
clustering_methods = {"KMeans": KMeans, "DBSCAN": DBSCAN, "Agglomerative": AgglomerativeClustering,
                          "Affinity Propagation": AffinityPropagation, "Spectral Clustering": SpectralClustering}

param_combinations_per_alg = {}
for key_alg in j_space:
    param_combinations_per_alg[key_alg] = list(product(*list(j_space[key_alg].values())))
    for param_combination in param_combinations_per_alg[key_alg]:
        trial_values = {}
        params = dict(zip(j_space[key_alg].keys(), list(param_combination)))
        labels_ = clustering_methods[key_alg](**params).fit_predict(x)

with open("results/es/kokoroko.json", "r") as f:
    es_results = json.load(f)

all_confs = []
for key in es_results["kokoroko"]:
    all_confs += es_results["kokoroko"][key]

min(all_confs, key=lambda x: x["cvi"]["silhouette"])
best_config_per_cvi_ = {}

sorted_configs = sorted(all_confs, key=lambda x: x["cvi"]["silhouette"])
diffs = [sorted_configs[i + 1]["cvi"]["silhouette"] - sorted_configs[i]["cvi"]["silhouette"] for i in range(len(sorted_configs) - 1)]
max_diff_index = diffs.index(max(diffs))
best_config_per_cvi[cvi] = sorted_configs[max_diff_index + 1]
