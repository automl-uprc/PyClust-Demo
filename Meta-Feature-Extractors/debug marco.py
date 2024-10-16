import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
from scipy.sparse import coo_matrix
import dgl
import torch

dataset = 'appendicitis'
dataset_path = os.path.join(r"D:\GitHub_D\dataset-exhaustive-search\datasets", dataset, "data.csv")
df = pd.read_csv(dataset_path, header=None)
df = PCA(n_components=0.9).fit_transform(df)
similarity_matrix = cosine_similarity(df)
similarity_matrix[similarity_matrix < 0.9] = 0

edge_list = []
src = []
dst = []
weight = []
threshold = 0.9
similarity_matrix[similarity_matrix < 0.9] = 0
for i in range(similarity_matrix.shape[0]):
    if np.count_nonzero(similarity_matrix[i]) == 1:
        similarity_matrix_ = np.delete(similarity_matrix, i,0)
        similarity_matrix_ = np.delete(similarity_matrix_, i, 1)
for i in range(similarity_matrix_.shape[0]):
    for c in range(i + 1, similarity_matrix_.shape[1]):
        val = similarity_matrix_[i, c]
        if val != 0:
            edge_list.append([" ".join([str(i), str(c), str(val)]) + "\n"])
            src.append(i)
            dst.append(c)
            weight.append(val)

path_to_write = fr"clustml\temp\edges\el_{dataset}.file"
with open(path_to_write, 'w') as f:
    for lis in edge_list:
        f.writelines(lis)

src_arr = np.array(src)
dst_arr = np.array(dst)
weight_arr = np.array(weight)
total_nodes = len(set(dst + src))
edge_coo = coo_matrix((weight_arr, (src_arr, dst_arr)), (total_nodes, total_nodes))


"""Part 2 """
edge_list_path = fr"clustml\temp\edges\el_{dataset}.file"
path_out = fr"clustml\temp\embeddings\ne_{dataset}.file"
cmd = 'deepwalk --input ' + edge_list_path + ' --format weighted_edgelist --output ' + path_out
cmd_output = os.popen(cmd).read()

"""Part 3"""
graph = dgl.from_scipy(edge_coo, eweight_name='weight')
graph = dgl.add_reverse_edges(graph, copy_edata=True)
graph = dgl.add_self_loop(graph)
if '_ID' in graph.edata:
    graph.edata.pop('_ID')

node_emb_path = rf"clustml\temp\embeddings\ne_{dataset}.file"
with open(node_emb_path, 'r') as f:
    line = f.readline()
    line = f.readline()
    node_feat_dict = {}
    while line:
        cur_line = line.split(" ")
        node_feat_dict[int(cur_line[0])] = [float(val) for val in cur_line[1:]]
        line = f.readline()

sorted_node_feat_dict = dict(sorted(node_feat_dict.items(), key=lambda e: e[0]))

node_feat_array = np.array(list(sorted_node_feat_dict.values()))

graph.ndata['h'] = torch.FloatTensor(node_feat_array)
model = torch.load(r'clustml/temp/gnn_model.pth')
model.eval()
prediction = model(graph, torch.rand((105, 64)))
