"""
This test includes the necessary code to compare the PyClust execution time with the python Library ClusterFeatures and
R library: clusterCrit. clusterCrit has been tested in native R and clusterFeatures was modified to be compatible with
the latest updates of its components.
"""
import matplotlib.pyplot as plt
from pyclust_eval import CVIToolbox
from sklearn.datasets import make_blobs
import time
from ClustersFeatures import *

n_samples_list = [1000, 5000, 10000, 20000]

exec_times_cluster_feat = []
for i in n_samples_list:
    x, y = make_blobs(n_samples=i, n_features=3)
    x = pd.DataFrame(x)
    x['target'] = y
    start_time = time.time()
    CC=ClustersCharacteristics(pd.DataFrame(x), label_target="target")
    asd = CC.IndexCore_compute_every_index()
    exec_time = time.time() - start_time
    exec_times_cluster_feat.append(exec_time)

exec_times = []
for i in n_samples_list:
    x, y = make_blobs(n_samples=i, n_features=3)
    cvit = CVIToolbox(x, y)
    start_time = time.time()
    cvit.calculate_icvi()
    exec_times.append(time.time() - start_time)


exec_times_r = [0.09410405,  3.72401905, 18.70204592, 93.09587502]
exec_times = [0.2044992446899414, 4.025498628616333, 16.740500688552856, 69.4884991645813]


"""
Below we visualize the results of the cvi calculations
exec_times_r = [0.09410405,  3.72401905, 18.70204592, 93.09587502]
exec_times = [0.2044992446899414, 4.025498628616333, 16.740500688552856, 69.4884991645813]
exec_times_cluster_feat = [18.619999647140503, 40.886000633239746, 89.54250168800354, 305.6704993247986]
"""

# Example data: lists of execution times
iterations = n_samples_list  # The number of iterations or budget

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(iterations, exec_times_cluster_feat, marker='o', linestyle='-', color='blue', label='ClusterFeatures')
plt.plot(iterations, exec_times_r, marker='o', linestyle='-', color='red', label='ClusterCrit')

plt.plot(iterations, exec_times, marker='o', linestyle='-', color='green', label='PyClust')

# Adding titles and labels
plt.title('Comparison of CVI Calculation Times Between Frameworks')
plt.xlabel('No_Samples')
plt.ylabel('Execution Time (seconds)')
plt.legend()

# Show grid for better readability
plt.grid(True)

# Show the plot
plt.savefig('exec_times_comparison.png')