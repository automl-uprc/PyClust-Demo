from sklearn.cluster import KMeans
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.vectors import IntVector
from rpy2.robjects.packages import importr
from pyclust_eval.utils.rpy import install_r_packages
from custom_cvis import *
from sklearn.datasets import make_blobs
from autometrics import AutoMetrics


def compute_internal_cvi(X: object, labels: object, cvi_engine="python", cvi_lib_loc=None):
    """Computes a set of internal cluster validation indices."""
    assert cvi_engine in ["r", "python"], "cvi_engine must be either python or r."

    # If the engine is se to python only the available indices of scikit learn will be computed.
    if cvi_engine == "python":
        int_cvi = {"silhouette": sc(X, labels),
                   "calinski_harabasz": chs(X, labels), "davies_bouldin": dbs(X, labels),
                   }

    # If the engine is set to r, a plethora of internal cluster validation indices will be computed according to
    # the clusterCrit package.
    elif cvi_engine == "r":
        rpy2.robjects.numpy2ri.activate()
        clusterCrit = ""
        fpc = ""

        # Import R libraries. The location of the r packages can be set manually, thus the parameter cvi_lib_loc
        # is to declare such path to packages.
        try:
            if cvi_lib_loc is not None:
                fpc = importr('fpc', lib_loc=cvi_lib_loc)
                clusterCrit = importr("clusterCrit", lib_loc=cvi_lib_loc)
            else:
                fpc = importr('fpc')
                clusterCrit = importr("clusterCrit")

        except Exception as e:
            install_r_packages(("fpc", "clusterCrit"), lib=cvi_lib_loc)

        base = importr("base")
        r_labels = base.match(labels, base.unique(labels))
        r_data = np.array(X)

        nr, nc = X.shape
        Br = ro.r.matrix(r_data, nrow=nr, ncol=nc)
        test = clusterCrit.intCriteria(Br, IntVector(r_labels), "all")
        int_cvi = dict(zip(test.names, map(list, list(test))))
        int_cvi = dict(map(lambda x: (x[0], float(x[1][0])), int_cvi.items()))

        cdbw = fpc.cdbw(Br, r_labels)[1][0]
        int_cvi["cdbw"] = cdbw

        # Correction for silhouette score.
        if str(int_cvi["silhouette"]) == "nan":
            int_cvi["silhouette"] = sc(X, labels)
    return int_cvi


x, y = make_blobs(n_samples=200, n_features=2, centers=5, random_state=666)
# x = MinMaxScaler().fit_transform(x)
km = KMeans().fit(x, y)

R_cvis = compute_internal_cvi(x, km.labels_, cvi_engine='r')

cvis_computed = [['trace_w', R_cvis['trace_w'], trace_w(x, km.labels_)],
                 ['mcclain_rao', R_cvis['mcclain_rao'], mcclain_rao(x, km.labels_)],
                 ['sd_dis', R_cvis['sd_dis'], sd_dis(x, km.labels_)],
                 ['sd_scat', R_cvis['sd_scat'], sd_scat(x, km.labels_)],
                 ['c_index', R_cvis['c_index'], c_index(x, km.labels_)],
                 ['banfeld_raftery', R_cvis['banfeld_raftery'], banfeld_raftery(x, km.labels_)],
                 ['det_ratio', R_cvis['det_ratio'], det_ratio(x, km.labels_)],
                 ['dunn', R_cvis['dunn'], dunn_index(x, km.labels_)],
                 ['gdi21', R_cvis['gdi21'], gdi21(x, km.labels_)],
                 ['gdi31', R_cvis['gdi31'], gdi31(x, km.labels_)],
                 ['gdi41', R_cvis['gdi41'], gdi41(x, km.labels_)],
                 ['gdi51', R_cvis['gdi51'], gdi51(x, km.labels_)],
                 ['gdi61', 'Cannot be computed', gdi61(x, km.labels_)],
                 ['gdi12', R_cvis['gdi12'], gdi12(x, km.labels_)],
                 ['gdi22', R_cvis['gdi22'], gdi22(x, km.labels_)],
                 ['gdi32', R_cvis['gdi32'], gdi32(x, km.labels_)],
                 ['gdi42', R_cvis['gdi42'], gdi42(x, km.labels_)],
                 ['gdi52', R_cvis['gdi52'], gdi52(x, km.labels_)],
                 ['gdi62', 'Cannot be computed', gdi62(x, km.labels_)],
                 ['gdi13', R_cvis['gdi13'], gdi13(x, km.labels_)],
                 ['gdi23', R_cvis['gdi23'], gdi23(x, km.labels_)],
                 ['gdi33', R_cvis['gdi33'], gdi33(x, km.labels_)],
                 ['gdi43', R_cvis['gdi43'], gdi43(x, km.labels_)],
                 ['gdi53', R_cvis['gdi53'], gdi53(x, km.labels_)],
                 ['gdi63', 'Cannot be computed', gdi63(x, km.labels_)],
                 ['pbm', R_cvis['pbm'], pbm(x, km.labels_)],
                 ['ratkowsky_lance', R_cvis['ratkowsky_lance'], ratkowsky_lance(x, km.labels_)],
                 ['ball_hall', R_cvis['ball_hall'], ball_hall(x, km.labels_)],
                 ['log_ss_ratio', R_cvis['log_ss_ratio'], log_ss_ratio(x, km.labels_)],
                 ['point_biserial', R_cvis['point_biserial'], point_biserial(x, km.labels_)],
                 ['ray_turi', R_cvis['ray_turi'], ray_turi(x, km.labels_)],
                 ['s_dbw', R_cvis['s_dbw'], s_dbw(x, km.labels_)],
                 ['wemmert_gancarski', R_cvis['wemmert_gancarski'], wemmert_gancarski(x, km.labels_)],
                 ['xie_beni', R_cvis['xie_beni'], xie_beni(x, km.labels_)],
                 ['cdbw', R_cvis['cdbw'], cdbw(x, km.labels_)],
                 ['log_ss_ratio', R_cvis['log_ss_ratio'], log_ss_ratio(x, km.labels_)],
                 ['ksq_detw', R_cvis['ksq_detw'], ksq_detw(x, km.labels_)],
                 ['log_det_ratio', R_cvis['log_det_ratio'], log_det_ratio(x, km.labels_)],
                 ['calinski_harabasz', R_cvis['calinski_harabasz'], calinski_harabasz(x, km.labels_)],
                 ['silhouette', R_cvis['silhouette'], silhouette(x, km.labels_)],
                 ['davies_bouldin', R_cvis['davies_bouldin'], davies_bouldin(x, km.labels_)],
                 ['friedman_rudin_1', 'Cannot be computed', friedman_rudin_1(x, km.labels_)],
                 ['friedman_rudin_2', 'Cannot be computed', friedman_rudin_2(x, km.labels_)]]

cvis_computed = pd.DataFrame(cvis_computed, columns=['CVI', 'R', 'Ours'])
cvis_computed = cvis_computed.set_index('CVI')
am = AutoMetrics(x, km.labels_).cvis
am = pd.DataFrame.from_dict(am, orient='index', columns=['AutoMetrics'])
df = pd.merge(cvis_computed, am, left_index=True, right_index=True)
print(df)

# Generate synthetic data
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import time
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist
from pyclust_eval.index_specifics.s import upper_triangle
import numpy as np
from custom_cvis import gamma, tau
from pyclust_eval import CVIToolbox

X, y = make_blobs(n_samples=150000, centers=4, cluster_std=0.60, random_state=0)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

start_time = time.time()
cvit = CVIToolbox(X,y)
cvit.calculate_icvi(["silhouette", "calinski_harabasz", "davies_bouldin"])
exec_time = time.time() - start_time

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
start_time = time.time()
sc = silhouette_score(X,y)
db = davies_bouldin_score(X,y)
ch = calinski_harabasz_score(X,y)
exec_time_ = time.time() - start_time

print(exec_time, exec_time_)

from scipy.spatial.distance import  cdist
cdist(X,X)

# -----------------------------------Test 1----------------------------------------------------------------------------
"""Test 1: pairwise distances exec time"""
start_time = time.time()
distances = pairwise_distances(X, metric='euclidean')
print(f"exec time: {time.time() - start_time}")

start_time = time.time()
distances_condensed = pdist(X, metric='euclidean')
print(f"exec time: {time.time() - start_time}")

# also test if pdist method of scipy returns with the same order the pairwise distances
x = upper_triangle(distances)

# -------------------------------Test 2 ----------------------------------------------------------------
"""Test 2: check gamma/tau functionality"""
start_time = time.time()
gamma_index = gamma(X, y_kmeans)
print(gamma_index)
print(f"exec time (gamma): {time.time() - start_time}")

start_time = time.time()
tau_index = tau(X, y_kmeans)
print(tau_index)
print(f"exec time (tau): {time.time() - start_time}")
