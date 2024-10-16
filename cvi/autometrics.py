import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from itertools import combinations, product
import math
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import calinski_harabasz_score as chs
from sklearn.metrics import silhouette_score as ss
from sklearn.metrics import davies_bouldin_score as dbs
import custom_cvis
from scipy.spatial import distance_matrix, distance
from inspect import getmembers, isfunction
import itertools

from pyclust_eval.index_specifics.s import return_s

def single_cvi(cvi_name: str, X: np.array, y: np.array):
    """ Calculates a given CVI that is implemented as part of the autometrics library."""
    implemented_cvi = [x[0] for x in getmembers(custom_cvis, isfunction)]
    assert cvi_name in implemented_cvi, "Selected CVI is not implemented or misspelled, please select one of the " \
                                        "following:\n{}".format(implemented_cvi)
    return getattr(custom_cvis, cvi_name)(X, y)


class AutoMetrics:
    def __init__(self, x, y, cvis=None):
        self.x = x
        self.y = y
        self.compute_basics()
        self.cvi_list = {'ball_hall': self.ball_hall, 'banfeld_raftery': self.banfeld_raftery, 'c_index': self.c_index,
                         'calinski_harabasz': self.calinski_harabasz, 'davies_bouldin': self.davies_bouldin,
                         'det_ratio': self.det_ratio, 'dunn': self.dunn, 'gdi12': self.gdi12, 'gdi13': self.gdi13,
                         'gdi21': self.gdi21, 'gdi22': self.gdi22, 'gdi23': self.gdi23, 'gdi31': self.gdi31,
                         'gdi32': self.gdi32, 'gdi33': self.gdi33, 'gdi41': self.gdi41, 'gdi42': self.gdi42,
                         'gdi43': self.gdi43, 'gdi51': self.gdi51, 'gdi52': self.gdi52, 'gdi53': self.gdi53,
                         'gdi61': self.gdi61, 'gdi62': self.gdi62, 'gdi63': self.gdi63, 'ksq_detw': self.ksq_detw,
                         'log_det_ratio': self.log_det_ratio, 'log_ss_ratio': self.log_ss_ratio,
                         'mcclain_rao': self.mcclain_rao, 'pbm': self.pbm, 'point_biserial': self.point_biserial,
                         'ray_turi': self.ray_turi, 'ratkowsky_lance': self.ratkowsky_lance,
                         'scott_symons': self.scott_symons, 'sd_scat': self.sd_scat, 'sd_dis': self.sd_dis,
                         's_dbw': self.s_dbw, 'silhouette': self.silhouette, 'trace_w': self.trace_w,
                         'wemmert_gancarski': self.wemmert_gancarski, 'xie_beni': self.xie_beni,
                         'friedman_rudin_1': self.friedman_rudin_1, 'friedman_rudin_2': self.friedman_rudin_2,
                         'trace_wib': self.trace_wib, #'cdbw': self.cdbw,
                         'gamma': self.gamma,'tau': self.tau}


        self.cvis = {}


    def compute_basics(self):
        self.s_plus, self.s_minus, self.nb, self.nw = return_s(self.x, self.y)

        # Find distinct clusters
        self.clusters = list(set(self.y))
        # Find centers
        self.centers = {i: self.x[self.y == i].mean(axis=0) for i in self.clusters}
        # Compute Z matrix
        self.Z = np.zeros((self.x.shape[0], len(self.clusters)))
        for i in range(self.x.shape[0]):
            self.Z[i, self.y[i]] = 1
        # Compute clusters' centers
        self.X_bar = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.Z.T, self.Z)), self.Z.T), self.x)
        # Compute between-cluster SSCP matrix
        self.B = np.matmul(np.matmul(np.matmul(self.X_bar.T, self.Z.T), self.Z), self.X_bar)
        # Compute total-sample sum-of-squares and crossproducts SSCP matrix
        self.T = np.matmul(self.x.T, self.x)
        # Compute within-cluster SSCP matrix
        self.W = self.T - self.B
        # Compute S_W and N_W
        self.S_W = 0
        self.N_W = 0
        for cluster in self.clusters:
            temp = self.x[self.y == cluster]
            for i in range(temp.shape[0] - 1):
                for j in range(i + 1, temp.shape[0]):
                    self.S_W += np.linalg.norm(temp[i] - temp[j])
                    self.N_W += 1
        # Compute S_B and N_B
        self.S_B = 0
        self.N_B = 0
        for cluster_i in range(len(self.clusters) - 1):
            temp_1 = self.x[self.y == self.clusters[cluster_i]]
            for i in temp_1:
                for cluster_l in range(cluster_i + 1, len(self.clusters)):
                    temp_2 = self.x[self.y == cluster_l]
                    for j in temp_2:
                        self.S_B += np.linalg.norm(i - j)
                        self.N_B += 1
        # Compute distances between centers
        self.centers_distances = {(i, j): np.linalg.norm(self.centers[i] - self.centers[j]) for i, j in
                                  list(product(self.clusters, self.clusters))}
        self.Deltas1 = {i: pairwise_distances(self.x[self.y == i], self.x[self.y == i]).max() for i in self.clusters}
        self.Deltas2 = {
            i: pdist(self.x[self.y == i]).sum() / (self.y[self.y == i].shape[0] * (self.y[self.y == i].shape[0] - 1))
            for i in self.clusters}
        self.Deltas3 = {
            i: np.linalg.norm(self.x[self.y == i] - self.centers[i], axis=1).sum() / (self.y[self.y == i].shape[0] / 2)
            for i in self.clusters}
        self.deltas1 = {(i, j): pairwise_distances(self.x[self.y == i], self.x[self.y == j]).min() for i, j in
                        combinations(self.clusters, r=2)}
        self.deltas2 = {(i, j): pairwise_distances(self.x[self.y == i], self.x[self.y == j]).max() for i, j in
                        combinations(self.clusters, r=2)}
        self.deltas3 = {(i, j): pairwise_distances(self.x[self.y == i], self.x[self.y == j]).sum() / (
                    self.y[self.y == i].shape[0] * self.y[self.y == j].shape[0]) for i, j in
                        combinations(self.clusters, r=2)}
        self.deltas4 = {(i, j): np.linalg.norm(self.centers[i] - self.centers[j]) for i, j in
                        combinations(self.clusters, r=2)}
        self.deltas5 = {(i, j): (np.linalg.norm(self.x[self.y == i] - self.centers[i], axis=1).sum() + np.linalg.norm(
            self.x[self.y == j] - self.centers[j], axis=1).sum()) / (
                                            self.y[self.y == i].shape[0] + self.y[self.y == j].shape[0]) for i, j in
                        combinations(self.clusters, r=2)}
        self.deltas6 = {(i, j): max(directed_hausdorff(self.x[self.y == i], self.x[self.y == j])[0],
                                    directed_hausdorff(self.x[self.y == j], self.x[self.y == i])[0]) for i, j in
                        combinations(self.clusters, r=2)}
        self.X_cen = self.x.mean(axis=0)
        self.BGSS = np.zeros((1, self.x.shape[1]))
        for j in range(self.BGSS.shape[0]):
            for cluster in self.clusters:
                self.BGSS[j] += self.y[self.y == cluster].shape[0] * (self.centers[cluster][j] - self.X_cen[j]) ** 2
        self.TSS = np.zeros((1, self.x.shape[1]))
        for j in range(self.TSS.shape[0]):
            for i in range(self.x.shape[0]):
                self.TSS[j] += (self.x[i, j] - self.X_cen[j]) ** 2

        self.WG_k = {
            i: np.array([np.linalg.norm(vector - self.centers[i]) ** 2 for vector in self.x[self.y == i]]).sum() for i
            in self.clusters}
        self.X_cl = {i: self.x[self.y == i] - self.centers[i] for i in self.clusters}
        self.WG_k_mat = {i: np.matmul(self.X_cl[i].T, self.X_cl[i]) for i in self.clusters}
        self.WGSS = np.array(list(self.WG_k.values())).sum()
        self.WG_mat = np.zeros((self.x.shape[1], self.x.shape[1]))
        for i in self.WG_k_mat.values():
            self.WG_mat += i
        self.X_ds = self.x - self.X_cen
        self.T_mat = np.matmul(self.X_ds.T, self.X_ds)
        self.B_mat = []
        for i in range(self.x.shape[0]):
            self.B_mat.append(self.centers[self.y[i]] - self.X_cen)
        self.B_mat = np.array(self.B_mat)
        self.BG_mat = np.matmul(self.B_mat.T, self.B_mat)
        print("AutoMetrics: Basics Computed")
    def gamma(self):
        try:
            return (self.s_plus - self.s_minus) / (self.s_plus + self.s_minus)
        except:
            return np.NaN
    def tau(self):
        try:
            result =  (self.s_plus - self.s_minus) / (self.nb * self.nw * ((self.x.shape[0] * (self.x.shape[0] - 1)) / 2)) * 1 / 2
            return np.NaN
        except:
            return np.NaN


    # def cdbw(self):
      #  reps = self.fft(self.x, self.y, 5)
       # creps = self.closest_representatives(self.x, reps)
       # rcr = self.respective_closest_representatives(reps, creps)
        #d, mp, stdev, pts_in = self.density(self.x, self.y, rcr)
        #idens = self.inter_density(d, len(np.unique(self.y)))
       # sep = self.cluster_separation(self.x, rcr, idens, 2)
       # intr_dens = self.intra_dens(self.x, self.y, reps, 0.1)
       # compactness, si_values = self.compactness(s_range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], X=self.x,
                                                #  y=self.y, reps=reps)
       # si_changes = self.intra_change(si_values)
       # cohesion = self.cohesion(compactness, si_changes)
       # sc = self.sc(compactness, sep)
       # return self.compute_cdbw(cohesion, sc)

    def change_y(self, new_labels):
        try:
            self.y = new_labels
            self.clusters = list(set(self.y))
        except:
            return np.nan

    def dunn(self):
        try:
            return min(self.deltas1.values())/max(self.Deltas1.values())
        except:
            return np.nan

    def trace_w(self):
        try:
            return self.W.trace()
        except:
            return np.nan


    def mcclain_rao(self):
        try:
            return (self.S_W / self.N_W) / (self.S_B / self.N_B)
        except:
            return np.nan

    def friedman_rudin_1(self):
        try:
            return (np.dot(np.linalg.inv(self.W), self.B)).trace()
        except:
            return np.nan

    def friedman_rudin_2(self):
        try:
            return np.linalg.norm(self.T) / np.linalg.norm(self.W)
        except:
            return np.nan

    def sd_dis(self):
        try:
            #Compute Dmax
            Dmax = np.array(list(self.centers_distances.values())).max()
            #Compute Dmin
            Dmin = np.delete(np.array(list(self.centers_distances.values())), np.where(np.array(list(self.centers_distances.values()))==0)).min()
            #Compute total separation between clusters
            Dis = 0
            for i in self.clusters:
                s = 0
                for j in self.clusters:
                    if i != j:
                        s += self.centers_distances[(i,j)]
                Dis += s**(-1)
            Dis = (Dmax / Dmin) * Dis
            return Dis
        except:
            return np.nan

    def sd_scat(self):
        try:
            #Compute clusters' standard deviation
            std = {i: np.sqrt(np.dot(np.var(self.x[self.y==i], 0), np.var(self.x[self.y==i], 0))) for i in self.clusters}
            #Return average scattering for clusters
            return np.array(list(std.values())).sum() / len(self.clusters) / np.sqrt(np.dot(np.var(self.x, 0), np.var(self.x, 0)))
        except:
            return np.nan

    def c_index(self):
        try:
            n_k = {i:self.y[self.y==i].shape[0] for i in self.clusters}
            N_w = int(np.array([n_k[i] * (n_k[i] - 1) / 2 for i in self.clusters]).sum())
            S_w = {i:pdist(self.x[self.y==i]).sum() for i in self.clusters}
            S_w = np.array([S_w[i] for i in self.clusters]).sum()
            dist = pdist(self.x)
            dist.sort()
            dist = np.array(dist)
            S_min = dist[:N_w].sum()
            S_max = dist[dist.shape[0]-N_w:].sum()
            return (S_w - S_min) / (S_max - S_min)
        except:
            return np.nan

    def banfeld_raftery(self):
        try:
            s = 0
            for cluster in self.clusters:
                s += self.y[self.y==cluster].shape[0] * np.log((np.linalg.norm(self.x[self.y==cluster] - self.centers[cluster], axis=1)**2).sum() / self.y[self.y==cluster].shape[0])
            return s
        except:
            return np.nan

    def gdi21(self):
        try:
            return min(self.deltas2.values())/max(self.Deltas1.values())
        except:
            return np.nan

    def gdi31(self):
        try:
            return min(self.deltas3.values())/max(self.Deltas1.values())
        except:
            return np.nan

    def gdi41(self):
        try:
            return min(self.deltas4.values())/max(self.Deltas1.values())
        except:
            return np.nan

    def gdi51(self):
        try:
            return min(self.deltas5.values())/max(self.Deltas1.values())
        except:
            return np.nan

    def gdi61(self):
        try:
            return min(self.deltas6.values())/max(self.Deltas1.values())
        except:
            return np.nan

    def gdi12(self):
        try:
            return min(self.deltas1.values())/max(self.Deltas2.values())
        except:
            return np.nan

    def gdi22(self):
        try:
            return min(self.deltas2.values())/max(self.Deltas2.values())
        except:
            return np.nan

    def gdi32(self):
        try:
            return min(self.deltas3.values())/max(self.Deltas2.values())
        except:
            return np.nan

    def gdi42(self):
        try:
            return min(self.deltas4.values())/max(self.Deltas2.values())
        except:
            return np.nan
        
    def gdi52(self):
        try:
            return min(self.deltas5.values())/max(self.Deltas2.values())
        except:
            return np.nan

    def gdi62(self):
        try:
            return min(self.deltas6.values())/max(self.Deltas2.values())
        except:
            return np.nan

    def gdi13(self):
        try:
            return min(self.deltas1.values())/max(self.Deltas3.values())
        except:
            return np.nan

    def gdi23(self):
        try:
            return min(self.deltas2.values())/max(self.Deltas3.values())
        except:
            return np.nan

    def gdi33(self):
        try:
            return min(self.deltas3.values())/max(self.Deltas3.values())
        except:
            return np.nan

    def gdi43(self):
        try:
            return min(self.deltas4.values())/max(self.Deltas3.values())
        except:
            return np.nan

    def gdi53(self):
        try:
            return min(self.deltas5.values())/max(self.Deltas3.values())
        except:
            return np.nan

    def gdi63(self):
        try:
            return min(self.deltas6.values())/max(self.Deltas3.values())
        except:
            return np.nan

    def pbm(self):
        try:
            D = {(i,j):np.linalg.norm(self.centers[i] - self.centers[j]) for i, j in combinations(self.clusters, r=2)}
            D_B = np.array(list(D.values())).max()
            E_W = 0
            for i in self.clusters:
                for vector in self.x[self.y==i]:
                    E_W += np.linalg.norm(vector - self.centers[i])
            E_T = np.linalg.norm(self.x - self.X_cen, axis=1).sum()
            return ((1/len(self.clusters)) * (E_T / E_W) * D_B)**2
        except:
            return np.nan

    def ratkowsky_lance(self):
        try:
            R_bar = (self.BGSS / self.TSS).mean()
            return math.sqrt(R_bar / len(self.clusters))
        except:
            return np.nan

    def ball_hall(self):
        try:
            sum_M_G = {i:np.array([np.linalg.norm(self.x[self.y==i] - self.centers[i])**2]).sum() for i in self.clusters}
            return np.array([sum_M_G[i] / self.y[self.y==i].shape[0] for i in self.clusters]).mean()
        except:
            return np.nan

    def ray_turi(self):
        try:
            Deltas = {(i,j):np.linalg.norm(self.centers[i] - self.centers[j])**2  for i, j in combinations(self.clusters, r=2)}
            return (self.WGSS / self.x.shape[0]) / np.array(list(Deltas.values())).min()
        except:
            return np.nan
    
    def scott_symons(self):
        try:
            return np.array([self.y[self.y==i].shape[0] * np.log(np.linalg.det(self.WG_k_mat[i] / self.y[self.y==i].shape[0])) for i in self.clusters]).sum()
        except:
            return np.nan

    def density(self, i, j):
        try:
            #Method used to compute s_dbw and cdbw
            dens = 0
            if i==j:
                center = self.centers[i]
                std = np.sqrt(np.dot(np.var(self.x[self.y==i], 0), np.var(self.x[self.y==i], 0)))
            else:
                center = (self.centers[i] + self.centers[j]) / 2
            if i==j:
                total = self.x[self.y==self.centers[i]].shape[0]
                vectors = self.x[self.y==self.centers[i]]
                std = (np.sqrt(np.dot(np.var(self.x[self.y==i], 0), np.var(self.x[self.y==i], 0))) + np.sqrt(np.dot(np.var(self.x[self.y==j], 0), np.var(self.x[self.y==j], 0)))) / 2
            else:
                total = self.x[self.y==self.centers[i]].shape[0] + self.x[self.y==self.centers[j]].shape[0]
                vectors = self.x[np.logical_or(self.y==self.centers[i], self.y==self.centers[j])]
            for v in range(total):
                if np.linalg.norm(vectors[v] - center) < std:
                    dens += 1
            return dens
        except:
            return np.nan
        
    def s_dbw(self):
        try:
            Density = 0
            for i in range(len(self.clusters)-1):
                for j in range(i+1, len(self.clusters)):
                    Density += self.density(self.clusters[i], self.clusters[j])
            Density *= 2 / (len(self.clusters) * (len(self.clusters) - 1))
            return self.sd_scat() + Density
        except:
            return np.nan

    def wemmert_gancarski(self):
        try:
            R = [np.linalg.norm(self.x[i] - self.centers[self.y[i]]) / np.array([np.linalg.norm(self.x[i] - self.centers[j]) for j in self.clusters if j!=self.y[i]]).min() for i in range(self.x.shape[0])]
            R = np.array(R)
            J_k = {i:max(0, 1- (R[self.y==i].sum() / self.y[self.y==i].shape[0])) for i in self.clusters}
            return np.array([self.y[self.y==i].shape[0] * J_k[i] for i in self.clusters]).sum() / self.x.shape[0]
        except:
            return np.nan

    def xie_beni(self):
        try:
            deltas = {(i,j):pairwise_distances(self.x[self.y==i], self.x[self.y==j]).min()**2  for i, j in combinations(self.clusters, r=2)}
            return self.WGSS / (np.array(list(deltas.values())).min() * self.x.shape[0])
        except:
            return np.nan

    def log_ss_ratio(self):
        try:
            WGSS = np.array([np.array(np.linalg.norm(self.x[self.y==cluster] - self.centers[cluster], axis=1)**2).sum() for cluster in self.clusters]).sum()
            BGSS = np.array([self.y[self.y==cluster].shape[0] * np.linalg.norm(self.centers[cluster] - self.X_cen)**2 for cluster in self.clusters]).sum()
            return np.log(BGSS / WGSS)
        except:
            return np.nan

    def det_ratio(self):
        try:
            return np.linalg.det(self.T_mat) / np.linalg.det(self.WG_mat)
        except:
            return np.nan

    def ksq_detw(self):
        try:
            return len(self.clusters)**2 * np.linalg.det(self.W)
        except:
            return np.nan

    def log_det_ratio(self):
        try:
            return self.x.shape[0] * np.log(np.linalg.det(self.T_mat) / np.linalg.det(self.WG_mat))
        except:
            return np.nan

    def point_biserial(self):
        try:
            return (self.S_W / self.N_W - self.S_B / self.N_B) * math.sqrt(self.N_W * self.N_B) / (self.x.shape[0] * (self.x.shape[0] - 1) / 2)
        except:
            return np.nan

    def calinski_harabasz(self):
        try:
            return chs(self.x, self.y)
        except:
            return np.nan


    def silhouette(self):
        try:
            return ss(self.x, self.y)
        except:
            return np.nan


    def davies_bouldin(self):
        try:
            return dbs(self.x, self.y)
        except:
            return np.nan
    
    
    def trace_wib(self):
        try:
            return (np.matmul(np.linalg.inv(self.WG_mat), self.BG_mat)).trace()
        except:
            return np.nan

    






    #Compute Inter_dens (inter-cluster density)
    def inter_density(self, densities, no_classes):
        max_dens_sum = 0
        for cluster_combination in densities:
            max_dens_sum += np.max(cluster_combination[1])
        return max_dens_sum * (1/ no_classes)


    #Compute Sep (cluster's separation)
    def cluster_separation(self, X, closest_reps, inter_dens, no_classes):
        distances = []
        for cluster_pair in closest_reps:
            for rcr in cluster_pair[1]:
                distances.append(distance.euclidean(X[rcr[0]], X[rcr[1]]))
        denominator = 1 + inter_dens
        return ( (1/no_classes) * np.min(distances) ) / denominator


    #Compute Intra_dens (relative intra-cluster density)
    def intra_dens(self, X, y, reps, s):
        total_card = []
        total_stdev = []
        for cluster in reps.keys():
            cluster_card = []
            X_temp = X[np.where(y==cluster)]
            radius = np.std(X_temp)
            total_stdev.append(radius)
            n_i = X_temp.shape[0]
            for cluster_rep in reps[cluster]:
                shrunk_rep = X[cluster_rep] - s
                dist_matrix = distance_matrix(np.reshape(shrunk_rep, (1,X.shape[1])), X_temp)
                rep_card = len(np.where(dist_matrix <= radius))
                cluster_card.append(rep_card)
            total_card.append(sum(cluster_card))
        dens_cl = sum(total_card) * (1/len(reps.keys()))
        return dens_cl / (len(np.unique(y) * sum(total_stdev)))


    #Compute Compactness
    def compactness(self, s_range, X,y,reps):
        total = []
        for i  in s_range:
            total.append(self.intra_dens(X,y, reps,i))

        return  sum(total) / len(s_range), total


    #Compute Intra_change (intra-density change)
    def intra_change(self, si_values):
        return  sum(np.ediff1d(si_values)) / (len(si_values) - 1)


    #Compute Cohesion
    def cohesion(self, compactness, intra_change):
        return compactness/ (1+ intra_change)


    #Compute SC Separation wrt Compactness)
    def sc(self, compactness, cluster_sep ):
        return compactness * cluster_sep
