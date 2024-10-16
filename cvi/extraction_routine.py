from autometrics import AutoMetrics
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, homogeneity_score, adjusted_mutual_info_score
from sklearn.metrics import completeness_score, fowlkes_mallows_score, homogeneity_completeness_v_measure
from sklearn.metrics import mutual_info_score, v_measure_score
import os, ast, re, sys
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
datasets_path = r'C:\Users\giann\OneDrive\Έγγραφα\GitHub\dataset-exhaustive-search\datasets'
results_path = r"C:\Users\giann\OneDrive\Έγγραφα\GitHub\dataset-exhaustive-search\results"
datasets = os.listdir(results_path)
am = AutoMetrics()

algorithms = ['AffinityPropagation', 'AgglomerativeClustering', 'Birch', 'DBSCAN', 'GMC', 'KHMeans',
              'MBKMeans', 'MeanShift', 'MinimumSpanningTree', 'OPTICS', 'SpectralClustering', 'KMeans']



X = pd.read_csv(r"C:\Users\giann\OneDrive\Έγγραφα\GitHub\dataset-exhaustive-search\results\appendicitis\preprocessed data.csv")
y_pred = pd.read_csv(r"C:\Users\giann\OneDrive\Έγγραφα\GitHub\dataset-exhaustive-search\results\appendicitis\AffinityPropagation.csv")
y_true = pd.read_csv(r"C:\Users\giann\OneDrive\Έγγραφα\GitHub\dataset-exhaustive-search\datasets\appendicitis\target.csv", header=None)
def results_conversion(df):
    df.Parameters = df.Parameters.apply(lambda x: ast.literal_eval(x))

    def add_comma(match):
        return match.group(0) + ','

    df['Clustering Labels'] = df['Clustering Labels'].apply(lambda x: re.sub(r'\[[0-9\.\s]+\]', add_comma, x))
    df['Clustering Labels'] = df['Clustering Labels'].apply(lambda x: re.sub(r'([0-9\.]+)', add_comma, x))
    df['Clustering Labels'] = df['Clustering Labels'].apply(lambda x: eval(x))
    df['Clustering Labels'] = df['Clustering Labels'].apply(lambda x: x[0] if type(x) == tuple else x)
    return df
le = LabelEncoder()
y_true = le.fit_transform(y_true)
y_pred = results_conversion(y_pred)
y_pred = y_pred["Clustering Labels"].iloc[0]


def compute_external_cvis(y_true: np.array, y_pred: np.array) -> pd.DataFrame:
    """Computes a series of external cluster validity indices implemented through sklearn.
    y_true : True cluster labels
    y_pred : Predicted cluster labels from any suitable clustering algorithm.
    """
    cvis = {}
    external_indices = ["adjusted_rand_score", "homogeneity_score", "adjusted_mutual_info_score", "completeness_score",
                        "fowlkes_mallows_score", "homogeneity_completeness_v_measure", "mutual_info_score",
                        "v_measure_score"]
    for external_index in external_indices:
        try:
            cvis[external_index.replace("_score", "")] = getattr(sklearn.metrics, external_indices[0])(y_true, y_pred)
        except:
            cvis[external_index.replace("_score", "")] = np.nan
    cvis = pd.DataFrame.from_dict(cvis, orient='index').T
    return cvis

compute_external_cvis(y_true, y_pred)













def cvis(X, cluster_labels, true_labels=None, metric='l2', external=False):
    internal_cvis = AutoMetrics(X.to_numpy(), LabelEncoder().fit_transform(np.array(cluster_labels))).cvis
    internal_cvis = pd.DataFrame.from_dict(internal_cvis, orient='index').T.reset_index()
    # internal_cvis = pd.DataFrame.from_dict(compute_internal_cvi(X, cluster_labels, cvi_engine='r'), orient='index').T.reset_index()
    if external:
        external_cvis = compute_external_cvis(true_labels, cluster_labels).reset_index()
        cvis_ = pd.concat([internal_cvis, external_cvis], axis=1).drop('index', axis=1)
        return cvis_
    else:
        return internal_cvis.drop('index', axis=1)





for dataset in datasets[:33]:
    print(dataset)
    x = pd.read_csv(results_path + '\\' + dataset + '\\preprocessed data.csv', header=None)
    if 'target.csv' in os.listdir(datasets_path + '\\' + dataset):
        external = True
        y = pd.read_csv(datasets_path + '\\' + dataset + '\\target.csv', header=None)
        y = list(LabelEncoder().fit_transform(y))
    else:
        external = False

    for algorithm in algorithms:
        results = pd.DataFrame()
        algorithm_result = pd.read_csv(results_path + '\\' + dataset + '\\' + algorithm + '.csv')
        algorithm_result = results_conversion(algorithm_result)

        if 'data' not in algorithm:
            print(algorithm)
            for i in range(algorithm_result.shape[0]):
                if external:
                    if 'metric' in algorithm_result.iloc[i].Parameters:
                        if algorithm_result.iloc[i].Parameters['metric'] == 'euclidean' or \
                                algorithm_result.iloc[i].Parameters['metric'] == 'l2':
                            cvis_results = cvis(x, algorithm_result.iloc[i]['Clustering Labels'], y,
                                                metric='l2', external=True)
                            passed = True
                    else:
                        cvis_results = cvis(x, algorithm_result.iloc[i]['Clustering Labels'], y,
                                            metric='l2', external=True)
                        passed = True
                else:
                    if 'metric' in algorithm_result.iloc[i].Parameters:
                        if algorithm_result.iloc[i].Parameters['metric'] == 'euclidean' or \
                                algorithm_result.iloc[i].Parameters['metric'] == 'l2':
                            cvis_results = cvis(x, algorithm_result.iloc[i]['Clustering Labels'],
                                                metric='l2', external=False)
                            passed = True
                    else:
                        cvis_results = cvis(x, algorithm_result.iloc[i]['Clustering Labels'], y,
                                            metric='l2', external=False)
                        passed = True

                if passed:
                    parameters = pd.DataFrame.from_dict(algorithm_result.iloc[i].Parameters, orient='index').T
                    cvis_results = pd.concat([parameters, cvis_results], axis=1)
                    cvis_results['Clustering Labels'] = str(algorithm_result.iloc[i]['Clustering Labels'])
                    results = pd.concat([results, cvis_results], axis=0)
                    passed = False

            try:
                os.mkdir(results_path + '\\' + dataset + '\\CVIs')
            except:
                pass
            results.to_csv(results_path + '\\' + dataset + '\\CVIs' + '\\' + algorithm + '.csv', index=False)
